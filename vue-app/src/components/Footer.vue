<template>
  <footer class="border-t border-gray-200 mt-12 py-6 text-center text-sm text-gray-500">
    <span>&copy; {{ year }} Qi Wang. Powered by Vue &amp; Vite.</span>
    <div class="flex justify-center space-x-4 mt-2">
      <a href="https://github.com/qw2261" target="_blank" class="hover:text-gray-700">GitHub</a>
      <a href="https://www.linkedin.com/in/qwangmatt/" target="_blank" class="hover:text-gray-700">LinkedIn</a>
      <a href="mailto:qw2261@columbia.edu" class="hover:text-gray-700">Email</a>
    </div>
  </footer>
</template>

<script setup>
const year = new Date().getFullYear()
</script>
