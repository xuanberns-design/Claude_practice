# Neon Snake

Classic Snake as a single HTML file with no dependencies.

## Play

Open `index.html` in a browser, or serve the folder:

```sh
python3 -m http.server -d snake 8000
# then visit http://localhost:8000
```

## Controls

- **Arrow keys / WASD** to steer (swipe on mobile)
- **Space** to start or pause

## Rules

- Eat the pink dots to grow; each one makes you slightly faster.
- Walls wrap around; biting yourself ends the game.
- Your best score is saved in the browser.
