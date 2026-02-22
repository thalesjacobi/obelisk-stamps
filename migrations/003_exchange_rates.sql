-- Exchange rate history for Obelisk Stamps
-- EUR is the base currency (rate = how many units of target per 1 EUR)
-- All historical rates are preserved; the latest rate per currency is the active one

CREATE TABLE IF NOT EXISTS exchange_rates (
  id INT AUTO_INCREMENT PRIMARY KEY,
  currency CHAR(3) NOT NULL,
  rate DECIMAL(12,6) NOT NULL,
  source VARCHAR(50) DEFAULT 'manual',
  fetched_at DATETIME NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_currency_fetched (currency, fetched_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Seed with current hardcoded values so the app works immediately
INSERT INTO exchange_rates (currency, rate, source, fetched_at)
VALUES
  ('USD', 1.080000, 'manual', NOW()),
  ('GBP', 0.860000, 'manual', NOW());
