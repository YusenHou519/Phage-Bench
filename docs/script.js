/* PhageBench project page — sticky-TOC scroll-spy + back-to-top.
   No dependencies. */
(function () {
    "use strict";

    var tocLinks = Array.prototype.slice.call(
        document.querySelectorAll(".contents nav a")
    );
    var sections = tocLinks
        .map(function (a) {
            var id = a.getAttribute("href").slice(1);
            return document.getElementById(id);
        })
        .filter(Boolean);

    var backToTop = document.querySelector(".back-to-top");

    function setActive(id) {
        tocLinks.forEach(function (a) {
            a.classList.toggle(
                "active-nav-item",
                a.getAttribute("href") === "#" + id
            );
        });
    }

    // Scroll-spy: highlight the section nearest the top of the viewport.
    function onScroll() {
        var marker = window.innerHeight * 0.28;
        var currentId = sections.length ? sections[0].id : null;
        for (var i = 0; i < sections.length; i++) {
            if (sections[i].getBoundingClientRect().top <= marker) {
                currentId = sections[i].id;
            }
        }
        if (currentId) setActive(currentId);

        if (backToTop) {
            backToTop.classList.toggle("visible", window.pageYOffset > 320);
        }
    }

    // Smooth-scroll for the back-to-top control.
    if (backToTop) {
        backToTop.addEventListener("click", function (e) {
            e.preventDefault();
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    }

    var ticking = false;
    window.addEventListener(
        "scroll",
        function () {
            if (!ticking) {
                window.requestAnimationFrame(function () {
                    onScroll();
                    ticking = false;
                });
                ticking = true;
            }
        },
        { passive: true }
    );

    onScroll();
})();
