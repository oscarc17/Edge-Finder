from __future__ import annotations

from typing import Any, Iterable

import requests

from polymarket_edge.config import settings


def _chunk(values: list[str], size: int) -> Iterable[list[str]]:
    for idx in range(0, len(values), size):
        yield values[idx : idx + size]


class PolymarketClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.timeout = settings.request_timeout_s

    def _request(
        self,
        method: str,
        base_url: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_payload: dict[str, Any] | list[Any] | None = None,
    ) -> Any:
        url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
        response = self.session.request(
            method=method,
            url=url,
            params=params,
            json=json_payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def fetch_markets(
        self,
        *,
        closed: bool | None = False,
        archived: bool | None = False,
        page_size: int = 500,
        max_rows: int | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            params: dict[str, Any] = {"limit": page_size, "offset": offset}
            if closed is not None:
                params["closed"] = str(closed).lower()
            if archived is not None:
                params["archived"] = str(archived).lower()
            batch = self._request("GET", settings.gamma_base_url, "/markets", params=params)
            if not batch:
                break
            rows.extend(batch)
            if max_rows is not None and len(rows) >= max_rows:
                return rows[:max_rows]
            if len(batch) < page_size:
                break
            offset += len(batch)
        return rows

    def fetch_active_markets(self, *, max_rows: int | None = None) -> list[dict[str, Any]]:
        return self.fetch_markets(closed=False, archived=False, max_rows=max_rows)

    def fetch_books(self, token_ids: list[str]) -> list[dict[str, Any]]:
        if not token_ids:
            return []
        books: list[dict[str, Any]] = []
        for subset in _chunk(token_ids, settings.books_chunk_size):
            payload = [{"token_id": token_id} for token_id in subset]
            batch = self._request("POST", settings.clob_base_url, "/books", json_payload=payload)
            if isinstance(batch, dict) and "data" in batch:
                books.extend(batch["data"])
            elif isinstance(batch, list):
                books.extend(batch)
        return books

    def fetch_prices(self, token_ids: list[str], side: str = "mid") -> Any:
        if not token_ids:
            return []
        payload = [{"token_id": token_id, "side": side} for token_id in token_ids]
        return self._request("POST", settings.clob_base_url, "/prices", json_payload=payload)

    def fetch_price_history(
        self,
        token_id: str,
        *,
        start_ts: int | None = None,
        end_ts: int | None = None,
        fidelity: int = 60,
    ) -> Any:
        params: dict[str, Any] = {"token_id": token_id, "fidelity": fidelity}
        if start_ts is not None:
            params["startTs"] = start_ts
        if end_ts is not None:
            params["endTs"] = end_ts
        return self._request("GET", settings.clob_base_url, "/prices-history", params=params)

    def fetch_simplified_markets(
        self,
        *,
        limit: int = 500,
        max_rows: int | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        next_cursor: str | None = None
        while True:
            params: dict[str, Any] = {"limit": limit}
            if next_cursor:
                params["next_cursor"] = next_cursor
            batch = self._request("GET", settings.clob_base_url, "/sampling-simplified-markets", params=params)
            if isinstance(batch, dict):
                data = batch.get("data") or batch.get("markets") or []
                next_cursor = batch.get("next_cursor") or batch.get("nextCursor")
            else:
                data = batch
                next_cursor = None
            if not data:
                break
            rows.extend(data)
            if max_rows is not None and len(rows) >= max_rows:
                return rows[:max_rows]
            if not next_cursor:
                break
        return rows

    def fetch_market_trades(
        self,
        market_id: str,
        *,
        limit: int = 500,
        offset: int = 0,
        order: str = "DESC",
    ) -> list[dict[str, Any]]:
        params = {"market": market_id, "limit": limit, "offset": offset, "sortBy": "TIMESTAMP", "sortDirection": order}
        data = self._request("GET", settings.data_base_url, "/trades", params=params)
        return data if isinstance(data, list) else data.get("data", [])

    def fetch_wallet_trades(
        self,
        wallet: str,
        *,
        limit: int = 500,
        offset: int = 0,
        order: str = "DESC",
    ) -> list[dict[str, Any]]:
        params = {"user": wallet, "limit": limit, "offset": offset, "sortBy": "TIMESTAMP", "sortDirection": order}
        data = self._request("GET", settings.data_base_url, "/trades", params=params)
        return data if isinstance(data, list) else data.get("data", [])

    def fetch_market_holders(
        self,
        condition_ids: list[str],
        *,
        limit: int = 20,
        min_balance: int = 1,
    ) -> list[dict[str, Any]]:
        if not condition_ids:
            return []
        params = {
            "market": ",".join(condition_ids),
            "limit": min(20, int(limit)),
            "minBalance": int(min_balance),
        }
        data = self._request("GET", settings.data_base_url, "/holders", params=params)
        return data if isinstance(data, list) else data.get("data", [])
