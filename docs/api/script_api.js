/**
 * API Documentation JavaScript
 * Extends the main site's script.js with API-specific functionality
 */

(function() {
  'use strict';

  // API Documentation Manager
  class APIDocumentation {
    constructor() {
      this.currentModule = null;
      this.searchIndex = [];
      this.modules = new Map();
      this.init();
    }

    init() {
      // Wait for DOM to be ready
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => this.setup());
      } else {
        this.setup();
      }
    }

    setup() {
      this.setupSearch();
      this.setupSidebarNavigation();
      this.setupLocalTOC();
      this.setupCodeEnhancements();
      this.setupScrollSpy();
      this.buildSearchIndex();
      this.setupModuleToggle();
    }

    // Search Functionality
    setupSearch() {
      const searchInput = document.querySelector('.api-search-input');
      if (!searchInput) return;

      // Create results container if it doesn't exist
      let searchResults = document.querySelector('.api-search-results');
      if (!searchResults) {
        searchResults = document.createElement('div');
        searchResults.className = 'api-search-results';
        searchInput.parentElement.appendChild(searchResults);
      }

      // Debounced search
      let searchTimeout;
      searchInput.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
          this.performSearch(e.target.value, searchResults);
        }, 300);
      });

      // Close search results when clicking outside
      document.addEventListener('click', (e) => {
        if (!searchInput.parentElement.contains(e.target)) {
          searchResults.style.display = 'none';
        }
      });

      // Handle search result clicks
      searchResults.addEventListener('click', (e) => {
        const resultItem = e.target.closest('.search-result-item');
        if (resultItem) {
          searchResults.style.display = 'none';
          searchInput.value = '';
        }
      });
    }

    performSearch(query, resultsContainer) {
      if (query.length < 2) {
        resultsContainer.style.display = 'none';
        return;
      }

      const results = this.searchIndex.filter(item => {
        const searchText = `${item.name} ${item.description} ${item.module}`.toLowerCase();
        return searchText.includes(query.toLowerCase());
      });

      this.displaySearchResults(results.slice(0, 10), resultsContainer, query);
    }

    displaySearchResults(results, container, query) {
      if (results.length === 0) {
        container.innerHTML = '<div class="no-results">No results found</div>';
      } else {
        const html = results.map(item => `
          <a href="#${item.id}" class="search-result-item">
            <div class="result-name">${this.highlightMatch(item.name, query)}</div>
            <div class="result-module">${item.module}</div>
          </a>
        `).join('');
        container.innerHTML = html;
      }
      container.style.display = 'block';
    }

    highlightMatch(text, query) {
      const regex = new RegExp(`(${query})`, 'gi');
      return text.replace(regex, '<strong>$1</strong>');
    }

    buildSearchIndex() {
      // Build search index from API modules
      document.querySelectorAll('.api-module').forEach(module => {
        const name = module.querySelector('h3 code')?.textContent || '';
        const description = module.querySelector('.api-module-description')?.textContent || '';
        const id = module.id || '';

        this.searchIndex.push({
          id,
          name,
          description,
          module: name
        });

        // Also index submodules
        module.querySelectorAll('.api-table tbody tr').forEach(row => {
          const cells = row.querySelectorAll('td');
          if (cells.length >= 2) {
            const submoduleName = cells[0].textContent.trim();
            const submoduleDesc = cells[1].textContent.trim();

            this.searchIndex.push({
              id,
              name: submoduleName,
              description: submoduleDesc,
              module: name
            });
          }
        });
      });
    }

    // Sidebar Navigation
    setupSidebarNavigation() {
      const sidebarLinks = document.querySelectorAll('.api-sidebar-nav a');

      sidebarLinks.forEach(link => {
        link.addEventListener('click', (e) => {
          const href = link.getAttribute('href');
          if (href && href.startsWith('#')) {
            e.preventDefault();

            // Update active state
            sidebarLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            // Smooth scroll to section
            const targetId = href.substring(1);
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
              const offset = 80; // Account for fixed header
              const elementPosition = targetElement.getBoundingClientRect().top;
              const offsetPosition = elementPosition + window.pageYOffset - offset;

              window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
              });

              // Update URL
              history.replaceState(null, '', href);
            }
          }
        });
      });
    }

    // Local Table of Contents
    setupLocalTOC() {
      const tocContainer = document.getElementById('local-toc');
      if (!tocContainer) return;

      const mainContent = document.querySelector('.api-main-content');
      if (!mainContent) return;

      // Find all module headers
      const headers = mainContent.querySelectorAll('.api-module-header h3');
      if (headers.length === 0) return;

      const tocList = document.createElement('ol');

      headers.forEach(header => {
        const module = header.closest('.api-module');
        if (!module || !module.id) return;

        const li = document.createElement('li');
        const link = document.createElement('a');
        link.href = `#${module.id}`;
        link.textContent = header.textContent.replace('svv.', '');

        // Handle click for smooth scrolling
        link.addEventListener('click', (e) => {
          e.preventDefault();
          const offset = 80;
          const elementPosition = module.getBoundingClientRect().top;
          const offsetPosition = elementPosition + window.pageYOffset - offset;

          window.scrollTo({
            top: offsetPosition,
            behavior: 'smooth'
          });

          history.replaceState(null, '', `#${module.id}`);
        });

        li.appendChild(link);
        tocList.appendChild(li);
      });

      tocContainer.appendChild(tocList);
    }

    // Code Enhancements
    setupCodeEnhancements() {
      // Copy button is already handled by the main script.js
      // Just ensure code blocks have the data-copy attribute
      document.querySelectorAll('.api-example pre').forEach(pre => {
        if (!pre.hasAttribute('data-copy')) {
          pre.setAttribute('data-copy', '');
        }
      });

      // Add language class for syntax highlighting
      document.querySelectorAll('.api-example code').forEach(code => {
        if (!code.className.includes('language-')) {
          code.className = 'language-python';
        }
      });
    }

    // Scroll Spy for Active Navigation
    setupScrollSpy() {
      const modules = document.querySelectorAll('.api-module');
      const sidebarLinks = document.querySelectorAll('.api-sidebar-nav a');
      const tocLinks = document.querySelectorAll('#local-toc a');

      if (modules.length === 0) return;

      const observerOptions = {
        root: null,
        rootMargin: '-20% 0px -70% 0px',
        threshold: 0
      };

      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const id = entry.target.getAttribute('id');

            // Update sidebar navigation
            sidebarLinks.forEach(link => {
              link.classList.remove('active');
              if (link.getAttribute('href') === `#${id}`) {
                link.classList.add('active');
              }
            });

            // Update local TOC
            tocLinks.forEach(link => {
              link.classList.remove('active');
              if (link.getAttribute('href') === `#${id}`) {
                link.classList.add('active');
              }
            });
          }
        });
      }, observerOptions);

      modules.forEach(module => {
        if (module.id) {
          observer.observe(module);
        }
      });
    }

    // Module Toggle (for mobile)
    setupModuleToggle() {
      const sidebar = document.querySelector('.api-sidebar');
      if (!sidebar) return;

      // Create toggle button for mobile
      const toggleBtn = document.createElement('button');
      toggleBtn.className = 'api-sidebar-toggle';
      toggleBtn.innerHTML = '<i class="fas fa-bars"></i> Modules';
      toggleBtn.style.display = 'none';

      // Insert before sidebar
      sidebar.parentElement.insertBefore(toggleBtn, sidebar);

      // Handle responsive behavior
      let resizeTimer;
      const checkResponsive = () => {
        const width = window.innerWidth;

        if (width <= 900) {
          toggleBtn.style.display = 'block';
          sidebar.style.display = 'none';
          sidebar.style.position = 'static';
        } else {
          toggleBtn.style.display = 'none';
          sidebar.style.display = 'block';
          sidebar.style.position = 'sticky';
          // Remove any mobile-specific inline styles
          sidebar.style.top = '';
          sidebar.style.left = '';
          sidebar.style.right = '';
          sidebar.style.bottom = '';
          sidebar.style.background = '';
          sidebar.style.zIndex = '';
          sidebar.style.padding = '';
          sidebar.style.overflowY = '';
        }

        // Force layout recalculation
        const grid = document.querySelector('.api-grid');
        if (grid) {
          grid.style.display = 'none';
          grid.offsetHeight; // Force reflow
          grid.style.display = '';
        }
      };

      // Toggle sidebar on mobile
      toggleBtn.addEventListener('click', () => {
        if (sidebar.style.display === 'none' || !sidebar.style.display) {
          sidebar.style.display = 'block';
          sidebar.style.position = 'fixed';
          sidebar.style.top = '0';
          sidebar.style.left = '0';
          sidebar.style.right = '0';
          sidebar.style.bottom = '0';
          sidebar.style.background = 'white';
          sidebar.style.zIndex = '1000';
          sidebar.style.padding = '2rem';
          sidebar.style.overflowY = 'auto';

          // Add close button
          const closeBtn = document.createElement('button');
          closeBtn.className = 'api-sidebar-close';
          closeBtn.innerHTML = '<i class="fas fa-times"></i>';
          closeBtn.style.position = 'absolute';
          closeBtn.style.top = '1rem';
          closeBtn.style.right = '1rem';
          closeBtn.style.background = 'none';
          closeBtn.style.border = 'none';
          closeBtn.style.fontSize = '1.5rem';
          closeBtn.style.cursor = 'pointer';

          closeBtn.addEventListener('click', () => {
            sidebar.style.display = 'none';
            closeBtn.remove();
          });

          sidebar.insertBefore(closeBtn, sidebar.firstChild);
        }
      });

      // Handle window resize with debouncing
      window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
          checkResponsive();

          // Remove mobile overlay if present
          const closeBtn = sidebar.querySelector('.api-sidebar-close');
          if (closeBtn && window.innerWidth > 900) {
            closeBtn.remove();
          }
        }, 250);
      });

      // Check on load
      checkResponsive();

      // Close sidebar when clicking a link on mobile
      sidebar.addEventListener('click', (e) => {
        if (e.target.tagName === 'A' && window.innerWidth <= 900) {
          sidebar.style.display = 'none';
          const closeBtn = sidebar.querySelector('.api-sidebar-close');
          if (closeBtn) closeBtn.remove();
        }
      });
    }

    // Smooth anchor scrolling with header offset
    setupAnchorScrolling() {
      document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
          const href = this.getAttribute('href');
          if (href && href !== '#') {
            const target = document.querySelector(href);
            if (target) {
              e.preventDefault();
              const offset = 80; // Account for fixed header
              const elementPosition = target.getBoundingClientRect().top;
              const offsetPosition = elementPosition + window.pageYOffset - offset;

              window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
              });

              history.replaceState(null, '', href);
            }
          }
        });
      });
    }
  }

  // Initialize the API Documentation
  const apiDoc = new APIDocumentation();

  // Export for potential external use
  window.APIDocumentation = APIDocumentation;

  // Additional utility functions
  window.apiUtils = {
    // Format code snippets
    formatCode: function(code, language = 'python') {
      return code.trim().replace(/\t/g, '  ');
    },

    // Copy code to clipboard (backup method)
    copyCode: function(element) {
      const code = element.textContent || element.innerText;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        return navigator.clipboard.writeText(code);
      } else {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = code;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        return Promise.resolve();
      }
    },

    // Generate unique ID for elements
    generateId: function(text) {
      return text.toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-|-$/g, '');
    }
  };

})();