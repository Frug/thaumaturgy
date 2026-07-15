"""Visual theme — matches the nicegui.io landing-page aesthetic.

Palette + fonts were lifted from the live nicegui.io styles, then the brand
blue was deepened for the dark theme (nicegui's own is the softer #5898d4):
  primary  #34618C (deep steel blue)  secondary #2AA198 (teal)
  accent   #f0a050 (warm orange)      muted text #9ba2ae
  light page #EDEFF3 / cards #fff    dark page #1A1D26 / cards #1e222c
  fonts: Inter (UI) + JetBrains Mono (code)

NOTE: fonts currently load from Google Fonts. For a fully-offline build we'll
self-host them later; kept as a CDN link for now to iterate on the look.
"""

from nicegui import ui

# Brand colors passed to Quasar via ui.colors() (called per page/client).
# Deep steel blue chosen to read well on the dark/night theme (white text on
# the active nav item stays high-contrast).
COLORS = dict(
    primary="#34618C",
    secondary="#2AA198",
    accent="#f0a050",
    dark="#1e222c",       # component/card surface in dark mode
    dark_page="#1A1D26",  # page background in dark mode
    positive="#3f9e5a",   # green for confirm/commit actions (Load, Save, ...)
    negative="#b83a3a",   # dark red for cancel/destructive actions (Unload, Reset, ...)
    info="#268BD2",
    warning="#f0a050",
)

_HEAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --tg-radius: 14px;
    --tg-muted: #9ba2ae;
    --tg-header-h: 56px;
  }

  body, .q-field, .q-btn, .q-item, .q-card {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
  .q-field--filled .q-field__control, code, pre, .font-mono {
    font-family: 'JetBrains Mono', ui-monospace, 'Fira Mono', monospace;
  }

  /* Page backgrounds */
  .body--light { background: #EDEFF3; color: #4a4f5a; }
  .body--dark  { background: #1A1D26; color: #EDEFF3; }

  /* Frosted, borderless header that blends into the page */
  .tg-header {
    height: var(--tg-header-h); min-height: var(--tg-header-h);
    background: rgba(237, 239, 243, 0.72) !important;
    backdrop-filter: saturate(180%) blur(12px);
    color: #4a4f5a !important;
    border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    box-shadow: none !important;
  }
  .body--dark .tg-header {
    background: rgba(26, 29, 38, 0.72) !important;
    color: #EDEFF3 !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  }

  /* Nav drawer */
  .tg-drawer { background: transparent !important; border-right: 1px solid rgba(0,0,0,0.06); }
  .body--dark .tg-drawer { border-right: 1px solid rgba(255,255,255,0.06); }
  .tg-nav-item { border-radius: 10px; transition: background 0.15s ease; }
  .tg-nav-item:hover { background: rgba(52, 97, 140, 0.14); }
  .tg-nav-item.tg-active { background: #34618C; color: #fff; }

  /* Compact chat-history list: square, tight rows that never overflow sideways */
  .tg-chat-item.q-item {
    border-radius: 0;
    min-height: 0;
    padding: 8px 8px 8px 14px;
    transition: background 0.15s ease;
  }
  .tg-chat-item:hover { background: rgba(52, 97, 140, 0.14); }
  .tg-chat-item.tg-active { background: #34618C; color: #fff; }
  .tg-chat-delete-section.q-item__section--side {
    padding-left: 4px;
    padding-right: 0;
  }
  .tg-chat-delete.q-btn {
    min-height: 24px;
    min-width: 24px;
    padding: 0;
  }
  .tg-chat-delete .q-icon { font-size: 16px; }
  /* Match the rounded container: round the first/last rows' outer corners */
  .tg-chat-list .tg-chat-item:first-child {
    border-top-left-radius: 8px; border-top-right-radius: 8px;
  }
  .tg-chat-list .tg-chat-item:last-child {
    border-bottom-left-radius: 8px; border-bottom-right-radius: 8px;
  }

  /* Soft, rounded cards with diffuse shadows */
  .q-card {
    border-radius: var(--tg-radius);
    box-shadow: 0 1px 3px rgba(20, 30, 55, 0.06), 0 8px 28px rgba(20, 30, 55, 0.05);
    border: 1px solid rgba(0, 0, 0, 0.04);
  }
  .body--dark .q-card {
    background: #1e222c;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.35), 0 8px 28px rgba(0, 0, 0, 0.28);
  }

  /* Form fields (selects, number inputs): darker-blue fill, rounded, no border */
  .tg-field .q-field__control {
    background: rgba(52, 97, 140, 0.16);
    border-radius: 10px;
  }
  .body--dark .tg-field .q-field__control {
    background: rgba(52, 97, 140, 0.30);
  }
  /* Kill Quasar's underline / outline pseudo-borders */
  .tg-field .q-field__control::before,
  .tg-field .q-field__control::after { border: none !important; }

  /* Snappier interactions — halve Quasar's default ~300ms transitions */
  .q-transition--fade-enter-active, .q-transition--fade-leave-active,
  .q-transition--scale-enter-active, .q-transition--scale-leave-active,
  .q-transition--jump-down-enter-active, .q-transition--jump-down-leave-active,
  .q-transition--jump-up-enter-active, .q-transition--jump-up-leave-active,
  .q-transition--slide-down-enter-active, .q-transition--slide-down-leave-active {
    transition-duration: 150ms !important;
    animation-duration: 150ms !important;
  }
  /* Rotating chevrons (select dropdown + expansion toggle) */
  .q-select__dropdown-icon,
  .q-expansion-item__toggle-icon { transition-duration: 150ms !important; }

  /* Right-side sliding info panel (scenario details, etc.) */
  .tg-slidepanel {
    position: fixed; top: var(--tg-header-h); right: 0;
    height: calc(100vh - var(--tg-header-h)); width: 340px;
    z-index: 60; transform: translateX(100%);
    transition: transform 150ms ease;
    background: #ffffff; border-left: 1px solid rgba(0, 0, 0, 0.08);
    box-shadow: -8px 0 24px rgba(20, 30, 55, 0.12);
    overflow-y: auto;
  }
  .body--dark .tg-slidepanel {
    background: #1e222c; border-left: 1px solid rgba(255, 255, 255, 0.06);
    box-shadow: -8px 0 24px rgba(0, 0, 0, 0.35);
  }
  .tg-slidepanel.tg-open { transform: translateX(0); }

  /* Transparent full-screen catcher: click anywhere off the panel to dismiss */
  .tg-backdrop { position: fixed; inset: var(--tg-header-h) 0 0 0; z-index: 55; display: none; }
  .tg-backdrop.tg-open { display: block; }

  /* Model-page filmstrip: panels Model(2/5) Params(2/5) SetsList(1/5) of a
     125%-wide strip. View shows Model+Params (50/50); edit shifts left 40% to
     show Params(1/2) + SetsList(1/4), with the remaining 1/4 empty. */
  .tg-strip { transition: transform 200ms ease; }
  .tg-strip.tg-edit { transform: translateX(-40%); }

  /* Grouped model-page controls. */
  .tg-pset-box {
    position: relative;
    border: 1px solid rgba(52, 97, 140, 0.55);
    border-radius: 12px;
    padding: 12px;
    background: rgba(52, 97, 140, 0.05);
  }
  .tg-server-output {
    height: 288px;
    min-height: 288px;  /* the card is a flex column; don't let it collapse */
    padding: 10px 12px;
    border-radius: 10px;
    background: rgba(52, 97, 140, 0.16);
  }
  .body--dark .tg-server-output {
    background: rgba(52, 97, 140, 0.30);
  }
  .tg-server-output-text {
    font-size: 12px;
    line-height: 1.45;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }

  /* Pills / badges: more breathing room */
  .q-badge { padding: 4px 10px; border-radius: 8px; font-weight: 500; }

  /* Rounded, calm buttons */
  .q-btn { border-radius: 10px; text-transform: none; font-weight: 500; }

  .text-muted { color: var(--tg-muted) !important; }
</style>
<script>
  // After choosing an item from a dropdown, drop focus so the field doesn't
  // stay visually "highlighted" (underline + lighter background). Scoped to
  // items inside popup menus, so nav items / lists are unaffected.
  document.addEventListener('click', function (e) {
    if (e.target.closest('.q-menu .q-item')) {
      setTimeout(function () {
        var el = document.activeElement;
        if (el && el.blur) el.blur();
      }, 0);
    }
  }, true);
</script>
"""


def head_html() -> str:
    """The <head> block (fonts + global CSS). Added once, globally."""
    return _HEAD_HTML


def apply_colors() -> None:
    """Apply brand colors for the current page/client. Call inside a page."""
    ui.colors(**COLORS)
