-- ============================================================
-- 台股訊號系統 — Supabase 建表 SQL
-- 在 Supabase Dashboard → SQL Editor 貼上執行
-- ============================================================

-- 1. 追蹤的股票清單
CREATE TABLE watchlist (
  id         SERIAL PRIMARY KEY,
  ticker     TEXT NOT NULL UNIQUE,   -- e.g. "2330.TW"
  name       TEXT NOT NULL,          -- e.g. "台積電"
  added_at   TIMESTAMPTZ DEFAULT NOW()
);

-- 2. 每次分析的訊號紀錄
CREATE TABLE signals (
  id              SERIAL PRIMARY KEY,
  ticker          TEXT NOT NULL,
  stock_name      TEXT NOT NULL,
  signal          TEXT NOT NULL,     -- "BUY" | "HOLD" | "SELL"
  confidence      TEXT,              -- "高" | "中" | "低"
  reason          TEXT,
  risk            TEXT,
  price           NUMERIC,
  tech_score      INT,
  fund_score      INT,
  tech_detail     TEXT,
  fund_detail     TEXT,
  news_summary    TEXT,
  analyzed_at     TIMESTAMPTZ DEFAULT NOW()
);

-- 3. 開放 RLS（讓 anon key 可讀寫）
ALTER TABLE watchlist ENABLE ROW LEVEL SECURITY;
ALTER TABLE signals   ENABLE ROW LEVEL SECURITY;

CREATE POLICY "allow_all" ON watchlist FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all" ON signals   FOR ALL USING (true) WITH CHECK (true);

-- 4. 預設追蹤幾支常見台股
INSERT INTO watchlist (ticker, name) VALUES
  ('2330.TW', '台積電'),
  ('2317.TW', '鴻海'),
  ('2454.TW', '聯發科'),
  ('2382.TW', '廣達'),
  ('2308.TW', '台達電');
