<template>
  <footer class="border-t border-gray-100 mt-16 py-8 text-center text-sm text-gray-400">
    <span>&copy; {{ year }} Qi Wang</span>
    <div class="flex justify-center space-x-6 mt-3">
      <a href="https://github.com/qw2261" target="_blank" class="hover:text-gray-700 transition-colors">GitHub</a>
      <a href="https://www.linkedin.com/in/qwangmatt/" target="_blank" class="hover:text-gray-700 transition-colors">LinkedIn</a>
      <a href="mailto:qw2261@columbia.edu" class="hover:text-gray-700 transition-colors">Email</a>
    </div>
  </footer>
</template>

<script setup>
const year = new Date().getFullYear()
</script>
