/**
 * @file    config.h
 * @brief   Project-wide path configuration.
 *
 * All paths are absolute to work regardless of working directory.
 * Adjust if running outside Docker.
 * @author  anshulbadhani
 * @date    01/05/2026
 */

#pragma once

#define DATA_DIR        "/app/data"
#define EMBEDDINGS_DIR  "/app/data/embeddings"