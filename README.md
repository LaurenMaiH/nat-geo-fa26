# laurenmaih.github.io

My personal site. Plain HTML and CSS, no build step, hosted free on GitHub Pages.

Everything sits in one folder, with no subfolders, so drag-and-drop uploads to GitHub keep every link working.
Keep it that way: put new files (posts, images, PDFs) next to `index.html`.

```
index.html               Home: intro, selected work, writing, contact
projects.html            All projects, with a copy-paste template for new ones
persona-vectors.html     Research write-up, hosted on the site
post-template.html       Starting point for new on-site posts
style.css                All styling. Colors and fonts are at the top.
favicon.svg              Browser tab icon
og.png                   Preview image shown when the link is shared
Lauren_Henderson_CV.pdf  The CV every "CV" link points to
```

---

## Deploy to GitHub Pages (about 10 minutes, no command line)

1. **Sign in** to GitHub as `LaurenMaiH`.
2. **Create the repo.** Click **+** (top right) → **New repository**.
   - Repository name: `LaurenMaiH.github.io` (exactly this; it's what makes the URL `laurenmaih.github.io`)
   - Visibility: **Public**
   - Leave "Add a README" unchecked (one is included here)
   - Click **Create repository**
3. **Upload the files.** On the new empty repo page, click **uploading an existing file**.
   Open the `site` folder on your computer, select **every file inside it** (Cmd+A / Ctrl+A), and drag them all into the browser.
   Drag the files, not the `site` folder itself.
   Click **Commit changes**.
4. **Turn on Pages.** Go to **Settings** → **Pages** (left sidebar).
   Under **Build and deployment**, set Source to **Deploy from a branch**, Branch to **main** and folder **/ (root)**, then **Save**.
   For a `username.github.io` repo this is often already set; if so, you're done here.
5. **Wait 1–2 minutes.** The **Actions** tab shows a "pages build and deployment" run; when it's green, open
   **https://laurenmaih.github.io**
6. **Check it**: open it on your phone, click every link, and download the CV once.
   To test the link preview, paste the URL into a message to yourself (Slack, iMessage).

### Prefer the command line?

```bash
cd site
git init -b main
git add .
git commit -m "First version of my site"
git remote add origin https://github.com/LaurenMaiH/LaurenMaiH.github.io.git
git push -u origin main
```
Then do step 4 above.

---

## Editing

**Preview locally:** double-click `index.html` to open it in your browser. Refresh after each change.
(Fonts load from Google, so you need to be online for them to look right.)

**Find content fast:** search the HTML for `EDIT:`. Those comments mark the tagline, bio, interests, and contact note.

**Publish a change:** either
- on github.com, open the file → pencil icon → edit → **Commit changes**, or
- drag the updated file into the repo with **Add file → Upload files** (it replaces the old one).

The live site updates within a minute or two.

### Swap in a new CV
Name your new PDF exactly `Lauren_Henderson_CV.pdf` and upload it, replacing the old one.
Every CV link on the site points to that file, so nothing else changes.

The current file is the one-page LaTeX version. Recompile, rename, re-upload.

### Add a project
1. Open `projects.html` and find the comment **HOW TO ADD A PROJECT**.
2. Copy the template inside it, paste it where it belongs in the list (newest first), and fill it in.
   Give it an `id` like `id="spotify-recs"` so you can link straight to it (`projects.html#spotify-recs`).
3. To feature it on the home page too, copy one of the `<li class="entry">` blocks under **SELECTED WORK** in `index.html` and edit it.

### Add a piece of writing
In `index.html`, find **WRITING**, copy one `<li class="post">` block to the top of the list, and change the link, title, one-line description, publication, and date.

- **Published elsewhere (Medium, etc.)?** Link straight to it.
- **Want it on your site?** Copy `post-template.html`, rename it (e.g. `my-post.html`), and follow the steps at the top of the file. Link the list entry to `my-post.html`. Images go next to it, too.
  Writing in Markdown is easier: ask Claude to convert it, or run `pandoc my-post.md -o body.html` and paste the result into the template.

### Change colors or fonts
Everything is in the **THEME** block at the top of `style.css`.
- Recolor accents: change `--accent` and `--link`.
- Dark mode only: delete the `@media (prefers-color-scheme: light)` block right below the theme.
  Right now the site follows the visitor's device setting (gruvbox dark or gruvbox light).
- New fonts: pick them on [Google Fonts](https://fonts.google.com), replace the `<link href="https://fonts.googleapis.com/...">` line in every HTML file, and update the `--font-*` lines.

---

## If you get a custom domain later

1. Buy the domain (Namecheap, Porkbun, Cloudflare, roughly $10–15/year for a `.com`).
2. In the repo: **Settings → Pages → Custom domain**, enter it, and follow GitHub's DNS instructions.
3. Tick **Enforce HTTPS** once it's available.
4. Update the site URL in the `og:url` and `og:image` lines of every HTML file.
   (`og.png` also shows `laurenmaih.github.io` in its text.)

The old `laurenmaih.github.io` link keeps working and redirects, so nothing you've already sent breaks.
