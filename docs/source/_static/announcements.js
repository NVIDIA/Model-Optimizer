document.addEventListener('DOMContentLoaded', () => {
  const search = document.querySelector('#announcement-search');
  const cards = Array.from(document.querySelectorAll('.announcement-card'));
  const tags = Array.from(document.querySelectorAll('.announcement-tag'));
  const empty = document.querySelector('#announcement-empty');
  let activeTag = 'all';

  if (!search || cards.length === 0) {
    return;
  }

  const update = () => {
    const query = search.value.trim().toLowerCase();
    let visible = 0;

    cards.forEach((card) => {
      const haystack = [card.dataset.title, card.dataset.summary, card.dataset.tags].join(' ').toLowerCase();
      const tagMatch = activeTag === 'all' || (card.dataset.tags || '').split(' ').includes(activeTag);
      const searchMatch = !query || haystack.includes(query);
      const show = tagMatch && searchMatch;
      card.hidden = !show;
      if (show) visible += 1;
    });

    if (empty) {
      empty.hidden = visible !== 0;
    }
  };

  tags.forEach((button) => {
    button.addEventListener('click', () => {
      activeTag = button.dataset.tag || 'all';
      tags.forEach((tag) => tag.classList.toggle('is-active', tag === button));
      update();
    });
  });

  search.addEventListener('input', update);
  update();
});
