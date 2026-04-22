-- Catalogue: visibility toggle. Items default to public; admins can unpublish them.

ALTER TABLE catalogue
  ADD COLUMN is_public TINYINT(1) NOT NULL DEFAULT 1;

CREATE INDEX idx_catalogue_public ON catalogue(is_public);
