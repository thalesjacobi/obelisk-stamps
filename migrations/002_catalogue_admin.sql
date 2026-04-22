-- Catalogue admin: add status for sold items, and a gallery table for multiple images per item.

-- Add status column to catalogue (existing rows default to 'available')
ALTER TABLE catalogue
  ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'available';

CREATE INDEX idx_catalogue_status ON catalogue(status);

-- Additional gallery images per catalogue item.
-- The primary/cover image remains in catalogue.image_url; this table holds extras.
CREATE TABLE IF NOT EXISTS catalogue_images (
  id INT AUTO_INCREMENT PRIMARY KEY,
  catalogue_id INT NOT NULL,
  image_url VARCHAR(500) NOT NULL,
  sort_order INT NOT NULL DEFAULT 0,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (catalogue_id) REFERENCES catalogue(id) ON DELETE CASCADE,
  INDEX idx_catalogue_images_catalogue (catalogue_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
