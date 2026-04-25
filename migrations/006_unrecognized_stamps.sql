-- Captures stamps detected by the ML API that could not be matched with
-- high confidence, so an admin can later enrich them and promote into
-- postbeeld_stamps.

CREATE TABLE IF NOT EXISTS unrecognized_stamps (
  id INT AUTO_INCREMENT PRIMARY KEY,
  image_url VARCHAR(1024) NULL,
  image_blob LONGBLOB NULL,
  detection_confidence DECIMAL(5,4) NULL,
  bbox_json VARCHAR(255) NULL,
  best_match_similarity DECIMAL(5,4) NULL,
  best_match_title VARCHAR(512) NULL,
  best_match_country VARCHAR(128) NULL,
  best_match_year INT NULL,
  user_id INT NULL,
  client_ip VARCHAR(64) NULL,
  title VARCHAR(512) NULL,
  country VARCHAR(128) NULL,
  year INT NULL,
  condition_text VARCHAR(128) NULL,
  price_value DECIMAL(10,2) NULL,
  price_currency CHAR(3) NULL,
  notes TEXT NULL,
  reviewed TINYINT(1) NOT NULL DEFAULT 0,
  promoted_stamp_id BIGINT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  reviewed_at DATETIME NULL,
  INDEX idx_unrecognized_reviewed (reviewed),
  INDEX idx_unrecognized_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
