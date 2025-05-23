document.addEventListener("DOMContentLoaded", function() {
    console.log("JavaScript Loaded!");
});
// Add any interactive JavaScript functionality here if needed
// For example, you can add animations or dynamic content updates

document.addEventListener('DOMContentLoaded', function() {
    // Example: Add a simple animation to the feature cards
    const featureCards = document.querySelectorAll('.feature-card');

    featureCards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'scale(1.05)';
            card.style.transition = 'transform 0.3s ease';
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'scale(1)';
        });
    });
});
document.addEventListener("DOMContentLoaded", function () {
    console.log("Page Loaded Successfully!");
});
// Initialize AOS (Animate on Scroll)
AOS.init({
    duration: 1000,
    easing: "ease-in-out",
});

// Typing Effect in Header
const text = "Welcome to Our Beautiful Website!";
let i = 0;
function typeEffect() {
    if (i < text.length) {
        document.getElementById("typing-text").innerHTML += text.charAt(i);
        i++;
        setTimeout(typeEffect, 100);
    }
}
typeEffect();

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Newsletter form submission
document.querySelector('.newsletter-form').addEventListener('submit', function (e) {
    e.preventDefault();
    const email = this.querySelector('input').value;
    alert(`Thank you for subscribing with ${email}!`);
    this.reset();
});

// JavaScript for FAQ Accordion
document.querySelectorAll('.faq-item h3').forEach(header => {
    header.addEventListener('click', () => {
        const faqItem = header.parentElement;
        faqItem.classList.toggle('active');
    });
});

// JavaScript for Count-Up Animation
document.addEventListener('DOMContentLoaded', () => {
    const stats = document.querySelectorAll('.stat h3');
    stats.forEach(stat => {
        const target = parseInt(stat.getAttribute('data-count'));
        let count = 0;
        const duration = 2000; // Animation duration in milliseconds
        const increment = target / (duration / 16); // 16ms per frame

        const updateCount = () => {
            count += increment;
            if (count < target) {
                stat.textContent = Math.ceil(count);
                requestAnimationFrame(updateCount);
            } else {
                stat.textContent = target;
            }
        };

        updateCount();
    });
});


(function () {
    let currentReviewIndex = 2; // Track the current review index

    // Function to show the review at a given index
    function showReview(index) {
        const reviews = document.querySelectorAll('.review-card');
        const totalReviews = reviews.length;

        // Loop the reviews (go to first review when at the end, or last review when at the beginning)
        if (index >= totalReviews) {
            currentReviewIndex = 0; // Loop back to the first review
        } else if (index < 0) {
            currentReviewIndex = totalReviews - 1; // Go to the last review
        } else {
            currentReviewIndex = index; // Update to the current review
        }

        // Scroll the reviews container horizontally
        const container = document.querySelector('.reviews-container');
        container.style.transform = `translateX(-${currentReviewIndex * 320}px)`; // Fixed syntax error
    }

    // Previous button action
    function prevReview() {
        showReview(currentReviewIndex - 1); // Show the previous review
    }

    // Next button action
    function nextReview() {
        showReview(currentReviewIndex + 1); // Show the next review
    }

    // Initialize by showing the first review
    document.addEventListener('DOMContentLoaded', function () {
        showReview(currentReviewIndex); // Show the first review when the page loads
    });

    // Expose the prevReview and nextReview functions to the global scope
    window.prevReview = prevReview;
    window.nextReview = nextReview;
})();