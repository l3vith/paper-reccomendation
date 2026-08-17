"""
Module for database operations.

This module provides a PaperStore class to manage SQLite storage for the scraped
research papers, handling inserts, queries, and JSONL exports.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperStore:
    """
    SQLite-based storage for research papers.
    """
    def __init__(self, db_path: str = 'data/papers.db'):
        """
        Initialize the PaperStore.

        Args:
            db_path (str): The path to the SQLite database file.
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Create the papers table if it does not exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS papers (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source TEXT,
                        external_id TEXT,
                        title TEXT,
                        abstract TEXT,
                        authors TEXT,
                        year INTEGER,
                        pdf_url TEXT,
                        categories TEXT,
                        citation_ids TEXT,
                        reference_ids TEXT,
                        scraped_at TEXT,
                        UNIQUE(source, external_id)
                    )
                ''')
                conn.commit()
                logger.info(f"Initialized database at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def insert_papers(self, papers: List[Dict[str, Any]]) -> int:
        """
        Insert papers into the database, ignoring conflicts.

        Args:
            papers (List[Dict[str, Any]]): List of paper dictionaries to insert.

        Returns:
            int: Number of papers successfully inserted.
        """
        inserted_count = 0
        now = datetime.utcnow().isoformat()
        
        query = '''
            INSERT INTO papers (
                source, external_id, title, abstract, authors, year, 
                pdf_url, categories, citation_ids, reference_ids, scraped_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source, external_id) DO NOTHING
        '''
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                for p in papers:
                    authors_json = json.dumps(p.get('authors', []))
                    categories_json = json.dumps(p.get('categories', []))
                    citation_ids_json = json.dumps(p.get('citation_ids', []))
                    reference_ids_json = json.dumps(p.get('reference_ids', []))
                    
                    cursor.execute(query, (
                        p.get('source'),
                        p.get('external_id'),
                        p.get('title'),
                        p.get('abstract'),
                        authors_json,
                        p.get('year'),
                        p.get('pdf_url'),
                        categories_json,
                        citation_ids_json,
                        reference_ids_json,
                        now
                    ))
                    if cursor.rowcount > 0:
                        inserted_count += 1
                        
                conn.commit()
                logger.info(f"Inserted {inserted_count} new papers (out of {len(papers)} provided).")
        except Exception as e:
            logger.error(f"Error inserting papers: {e}")
            
        return inserted_count

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a sqlite3.Row into a dictionary, parsing JSON fields."""
        paper = dict(row)
        for field in ['authors', 'categories', 'citation_ids', 'reference_ids']:
            if paper.get(field):
                try:
                    paper[field] = json.loads(paper[field])
                except json.JSONDecodeError:
                    paper[field] = []
            else:
                paper[field] = []
        return paper

    def get_all_papers(self) -> List[Dict[str, Any]]:
        """
        Retrieve all papers from the database.

        Returns:
            List[Dict[str, Any]]: List of all paper dictionaries.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM papers")
                rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error retrieving all papers: {e}")
            return []

    def get_paper_by_id(self, paper_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single paper by its primary key.

        Args:
            paper_id (int): The primary key ID.

        Returns:
            Optional[Dict[str, Any]]: The paper dictionary, or None if not found.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
                row = cursor.fetchone()
                return self._row_to_dict(row) if row else None
        except Exception as e:
            logger.error(f"Error retrieving paper by id {paper_id}: {e}")
            return None

    def get_papers_with_citations(self) -> List[Dict[str, Any]]:
        """
        Retrieve papers that have citations or references.

        Returns:
            List[Dict[str, Any]]: List of papers with non-empty citations or references.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Assuming empty lists are stored as '[]'
                query = '''
                    SELECT * FROM papers 
                    WHERE citation_ids != '[]' 
                       OR reference_ids != '[]'
                       OR (citation_ids IS NOT NULL AND citation_ids != '[]')
                '''
                cursor.execute(query)
                rows = cursor.fetchall()
                
                papers = []
                for row in rows:
                    p = self._row_to_dict(row)
                    if p.get('citation_ids') or p.get('reference_ids'):
                        papers.append(p)
                return papers
        except Exception as e:
            logger.error(f"Error retrieving papers with citations: {e}")
            return []

    def export_jsonl(self, output_path: str = 'data/papers.jsonl') -> None:
        """
        Export all papers to a JSONL file.

        Args:
            output_path (str): Path to the output JSONL file.
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            papers = self.get_all_papers()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for paper in papers:
                    f.write(json.dumps(paper) + '\n')
                    
            logger.info(f"Exported {len(papers)} papers to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting to JSONL: {e}")

    def count(self) -> int:
        """
        Count the total number of papers in the database.

        Returns:
            int: The total count of papers.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM papers")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error counting papers: {e}")
            return 0
