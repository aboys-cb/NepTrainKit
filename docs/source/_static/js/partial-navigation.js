(function () {
  "use strict";

  const contentSelector = ".wy-nav-content";
  const menuSelector = ".wy-menu-vertical";
  const sideScrollSelector = ".wy-side-scroll";

  function injectCurrentPageToc() {
    const menu = document.querySelector(menuSelector);
    if (!menu) {
      return;
    }
    menu.querySelectorAll(".page-local-toc").forEach((node) => node.remove());

    const active = menu.querySelector("a.current") || menu.querySelector("li.current > a[href]");
    const activeItem = active ? active.closest("li") : null;
    const headings = Array.from(document.querySelectorAll(".rst-content section[id] > h2"));
    if (!activeItem || headings.length === 0) {
      return;
    }

    const activeLevel = Array.from(activeItem.classList)
      .map((name) => /^toctree-l(\d+)$/.exec(name))
      .find(Boolean);
    const childLevel = activeLevel ? Math.min(Number(activeLevel[1]) + 1, 6) : 2;
    const list = document.createElement("ul");
    list.className = "current page-local-toc";
    headings.forEach((heading) => {
      const item = document.createElement("li");
      item.className = `toctree-l${childLevel}`;
      const link = document.createElement("a");
      link.className = "reference internal";
      link.href = `#${heading.parentElement.id}`;
      link.textContent = heading.textContent.replace("", "").trim();
      item.appendChild(link);
      list.appendChild(item);
    });
    activeItem.appendChild(list);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", injectCurrentPageToc, { once: true });
  } else {
    injectCurrentPageToc();
  }

  if (!window.fetch || !window.DOMParser || window.location.protocol === "file:") {
    return;
  }

  function urlFor(anchor, baseUrl) {
    return new URL(anchor.getAttribute("href"), baseUrl);
  }

  function isDocumentUrl(url) {
    return (
      url.origin === window.location.origin &&
      (url.pathname.endsWith(".html") || url.pathname.endsWith("/"))
    );
  }

  function normalizeMenuLinks(menu, baseUrl) {
    menu.querySelectorAll("a[href]").forEach((anchor) => {
      const raw = anchor.getAttribute("href");
      if (!raw || raw.startsWith("javascript:")) {
        return;
      }
      try {
        const resolved = new URL(raw, baseUrl);
        if (resolved.origin === window.location.origin) {
          anchor.setAttribute("href", resolved.href);
        }
      } catch (_error) {
        // Leave malformed or non-URL theme actions untouched.
      }
    });
  }

  function pathKey(url) {
    const value = new URL(url, window.location.href);
    return value.pathname.replace(/\/index\.html$/, "/");
  }

  function matchingAnchor(menu, url, baseUrl) {
    const target = pathKey(url);
    return Array.from(menu.querySelectorAll("a[href]")).find((anchor) => {
      try {
        return pathKey(urlFor(anchor, baseUrl)) === target;
      } catch (_error) {
        return false;
      }
    });
  }

  function directChild(item, selector) {
    return Array.from(item ? item.children : []).find((child) =>
      child.matches(selector)
    );
  }

  function smallestSharedBranch(menu, nextTarget, targetUrl) {
    let nextItem = nextTarget ? nextTarget.closest("li") : null;
    while (nextItem) {
      const nextAnchor = directChild(nextItem, "a[href]");
      if (nextAnchor) {
        const currentAnchor = matchingAnchor(
          menu,
          urlFor(nextAnchor, targetUrl),
          window.location.href
        );
        if (currentAnchor) {
          return {
            currentItem: currentAnchor.closest("li"),
            nextItem,
          };
        }
      }
      const parentList = nextItem.parentElement;
      nextItem = parentList ? parentList.closest("li") : null;
    }
    return null;
  }

  function updateMenu(nextDocument, targetUrl) {
    const menu = document.querySelector(menuSelector);
    const nextMenu = nextDocument.querySelector(menuSelector);
    if (!menu || !nextMenu) {
      return;
    }

    normalizeMenuLinks(menu, window.location.href);
    normalizeMenuLinks(nextMenu, targetUrl);

    const nextTarget = matchingAnchor(nextMenu, targetUrl, targetUrl);
    const sharedBranch = smallestSharedBranch(menu, nextTarget, targetUrl);
    if (sharedBranch) {
      const currentList = directChild(sharedBranch.currentItem, "ul");
      const nextList = directChild(sharedBranch.nextItem, "ul");
      if (nextList && (!currentList || currentList.innerHTML !== nextList.innerHTML)) {
        if (currentList) {
          currentList.replaceWith(nextList.cloneNode(true));
        } else {
          sharedBranch.currentItem.appendChild(nextList.cloneNode(true));
        }
      }
    } else {
      menu.innerHTML = nextMenu.innerHTML;
    }

    menu.querySelectorAll(".current").forEach((node) => node.classList.remove("current"));
    const active = matchingAnchor(menu, targetUrl, targetUrl);
    if (!active) {
      return;
    }
    active.classList.add("current");
    let item = active.closest("li");
    while (item && menu.contains(item)) {
      item.classList.add("current");
      const list = item.parentElement;
      if (list) {
        list.classList.add("current");
      }
      item = list ? list.closest("li") : null;
    }
  }

  async function refreshDynamicFeatures(content) {
    if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
      try {
        await window.MathJax.typesetPromise([content]);
      } catch (error) {
        console.error("NepTrainKit docs: MathJax refresh failed", error);
      }
    }
    if (typeof window.addCopyButtonToCodeCells === "function") {
      window.addCopyButtonToCodeCells();
    }
    document.dispatchEvent(new CustomEvent("neptrainkit:page-swapped"));
  }

  async function navigate(target, pushState) {
    const url = new URL(target, window.location.href);
    const currentContent = document.querySelector(contentSelector);
    if (!currentContent) {
      window.location.assign(url.href);
      return;
    }

    document.documentElement.classList.add("docs-partial-loading");
    const sideScroll = document.querySelector(sideScrollSelector);
    const sideScrollTop = sideScroll ? sideScroll.scrollTop : 0;

    try {
      const response = await fetch(url.href, {
        headers: { "X-NepTrainKit-Partial-Navigation": "1" },
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const nextDocument = new DOMParser().parseFromString(await response.text(), "text/html");
      const nextContent = nextDocument.querySelector(contentSelector);
      if (!nextContent) {
        throw new Error("Target page has no documentation content");
      }

      currentContent.replaceChildren(...Array.from(nextContent.childNodes));
      document.title = nextDocument.title;
      updateMenu(nextDocument, url.href);
      injectCurrentPageToc();
      if (pushState) {
        window.history.pushState({ partialNavigation: true }, "", url.href);
      }
      if (sideScroll) {
        sideScroll.scrollTop = sideScrollTop;
      }

      if (url.hash) {
        const targetNode = document.getElementById(decodeURIComponent(url.hash.slice(1)));
        if (targetNode) {
          targetNode.scrollIntoView();
        }
      } else {
        window.scrollTo({ top: 0, left: 0, behavior: "auto" });
      }
      await refreshDynamicFeatures(currentContent);
    } catch (error) {
      console.error("NepTrainKit docs: partial navigation failed", error);
      window.location.assign(url.href);
    } finally {
      document.documentElement.classList.remove("docs-partial-loading");
    }
  }

  normalizeMenuLinks(document.querySelector(menuSelector) || document, window.location.href);

  document.addEventListener("click", (event) => {
    if (
      event.defaultPrevented ||
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey
    ) {
      return;
    }
    const eventElement =
      event.target instanceof Element ? event.target : event.target.parentElement;
    const anchor = eventElement ? eventElement.closest("a[href]") : null;
    if (!anchor || anchor.target || anchor.hasAttribute("download")) {
      return;
    }
    const raw = anchor.getAttribute("href");
    if (!raw || raw.startsWith("#")) {
      return;
    }
    let url;
    try {
      url = new URL(raw, window.location.href);
    } catch (_error) {
      return;
    }
    if (!isDocumentUrl(url)) {
      return;
    }
    if (pathKey(url) === pathKey(window.location.href) && url.hash) {
      return;
    }
    event.preventDefault();
    navigate(url.href, true);
  });

  window.addEventListener("popstate", () => navigate(window.location.href, false));
})();
