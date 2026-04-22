-- Catalogue: key/value specs per item (e.g. Dimensions, Stamps, Era, Origin).

CREATE TABLE IF NOT EXISTS catalogue_specs (
  id INT AUTO_INCREMENT PRIMARY KEY,
  catalogue_id INT NOT NULL,
  label VARCHAR(100) NOT NULL,
  value VARCHAR(500) NOT NULL,
  sort_order INT NOT NULL DEFAULT 0,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (catalogue_id) REFERENCES catalogue(id) ON DELETE CASCADE,
  INDEX idx_catalogue_specs_catalogue (catalogue_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
