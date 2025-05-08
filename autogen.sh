#!/bin/sh
# Run this to generate all the initial makefiles, etc.

# Create m4 directory if it doesn't exist
mkdir -p m4

autoreconf --force --install

echo "Now run ./configure and make"
