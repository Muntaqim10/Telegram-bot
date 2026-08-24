import json
import logging
import asyncio
import aiohttp
from typing import List

log = logging.getLogger(__name__)

class TradierMarketStream:
    """
    Connects to Tradier's WebSocket API for real-time market data ticks.
    Pushes valid price/volume events into an asyncio.Queue for the execution engine.
    """
    def __init__(self, token: str, symbols: List[str], event_queue: asyncio.Queue):
        self.token = token
        self.symbols = symbols
        self.event_queue = event_queue
        self.session_id = None
        self.ws = None
        self._running = False
        
    async def _create_session(self, session: aiohttp.ClientSession) -> str:
        """Fetch a new WebSocket session ID from Tradier REST API."""
        url = "https://api.tradier.com/v1/markets/events/session"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        }
        async with session.post(url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("stream", {}).get("sessionid")
            else:
                text = await resp.text()
                log.error(f"Failed to create Tradier session: {resp.status} - {text}")
                return None

    async def run(self):
        self._running = True
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    # 1. Get Session ID
                    self.session_id = await self._create_session(session)
                    if not self.session_id:
                        await asyncio.sleep(5)
                        continue
                        
                    log.info("Tradier WS session created. Connecting stream...")
                    
                    # 2. Connect to WebSocket
                    ws_url = "wss://ws.tradier.com/v1/markets/events"
                    async with session.ws_connect(ws_url) as ws:
                        self.ws = ws
                        # 3. Subscribe to symbols
                        payload = {
                            "sessionid": self.session_id,
                            "symbols": self.symbols,
                            "filter": ["trade", "quote"],
                            "linebreak": True,
                            "validOnly": True
                        }
                        await ws.send_json(payload)
                        log.info(f"Subscribed to {len(self.symbols)} symbols on Tradier.")
                        
                        # 4. Listen for events
                        async for msg in ws:
                            if not self._running:
                                break
                                
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                    # Tradier sends {"type": "trade", "symbol": "AAPL", "price": 150.0, "size": 100, ...}
                                    if data.get("type") in ("trade", "quote", "summary"):
                                        await self.event_queue.put(data)
                                except json.JSONDecodeError:
                                    pass
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                log.error(f"Tradier WS Connection Error: {ws.exception()}")
                                break
                                
            except Exception as e:
                log.error(f"Tradier Stream error: {e}")
                
            if self._running:
                log.warning("Tradier Stream disconnected. Reconnecting in 3 seconds...")
                await asyncio.sleep(3)
                
    async def update_symbols(self, new_symbols: List[str]):
        """Dynamically updates active symbols on Tradier stream."""
        old_count = len(self.symbols)
        self.symbols = new_symbols
        
        if getattr(self, 'ws', None) and not self.ws.closed and self.session_id:
            payload = {
                "sessionid": self.session_id,
                "symbols": self.symbols,
                "filter": ["trade", "quote"],
                "linebreak": True,
                "validOnly": True
            }
            try:
                await self.ws.send_json(payload)
                log.info(f"🔄 [DYNAMIC STREAM] Symbol subscription live-updated from {old_count} to {len(self.symbols)} tickers without reconnecting.")
            except Exception as e:
                log.warning(f"🔄 [DYNAMIC STREAM] Failed to live-update symbols (will retry on next reconnect): {e}")
        else:
            log.info(f"🔄 [DYNAMIC STREAM] Symbol subscription pool updated from {old_count} to {len(self.symbols)} tickers (deferred to next reconnect).")

    def stop(self):
        self._running = False
