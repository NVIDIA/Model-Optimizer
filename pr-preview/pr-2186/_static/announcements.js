document.addEventListener('DOMContentLoaded', () => {
  const search = document.querySelector('#announcement-search');
  const cards = Array.from(document.querySelectorAll('.announcement-card')).sort((left, right) => {
    return (right.dataset.date || '').localeCompare(left.dataset.date || '');
  });
  const tags = Array.from(document.querySelectorAll('.announcement-tag'));
  const empty = document.querySelector('#announcement-empty');
  const pager = document.querySelector('#announcement-pager');
  const prev = document.querySelector('#announcement-prev');
  const next = document.querySelector('#announcement-next');
  const status = document.querySelector('#announcement-page-status');
  const pageSize = 5;

  cards.forEach((card) => card.parentNode.appendChild(card));
  let activeTag = 'all';
  let currentPage = 1;

  if (!search || cards.length === 0) {
    return;
  }

  const matchingCards = () => {
    const query = search.value.trim().toLowerCase();
    return cards.filter((card) => {
      const haystack =
          [ card.dataset.title, card.dataset.summary, card.dataset.tags ].join(' ').toLowerCase();
      const tagMatch =
          activeTag === 'all' || (card.dataset.tags || '').split(' ').includes(activeTag);
      const searchMatch = !query || haystack.includes(query);
      return tagMatch && searchMatch;
    });
  };

  const update = () => {
    const matches = matchingCards();
    const pageCount = Math.max(1, Math.ceil(matches.length / pageSize));
    currentPage = Math.min(currentPage, pageCount);
    const start = (currentPage - 1) * pageSize;
    const pageCards = new Set(matches.slice(start, start + pageSize));

    cards.forEach((card) => { card.hidden = !pageCards.has(card); });

    if (empty) {
      empty.hidden = matches.length !== 0;
    }

    if (pager && prev && next && status) {
      pager.hidden = matches.length <= pageSize;
      prev.disabled = currentPage <= 1;
      next.disabled = currentPage >= pageCount;
      status.textContent = `Page ${currentPage} of ${pageCount}`;
    }
  };

  tags.forEach((button) => {
    button.addEventListener('click', () => {
      activeTag = button.dataset.tag || 'all';
      currentPage = 1;
      tags.forEach((tag) => {
        const selected = tag === button;
        tag.classList.toggle('is-active', selected);
        tag.setAttribute('aria-pressed', selected ? 'true' : 'false');
      });
      update();
    });
  });

  search.addEventListener('input', () => {
    currentPage = 1;
    update();
  });

  if (prev) {
    prev.addEventListener('click', () => {
      currentPage -= 1;
      update();
    });
  }

  if (next) {
    next.addEventListener('click', () => {
      currentPage += 1;
      update();
    });
  }

  update();
});
