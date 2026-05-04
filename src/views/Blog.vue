<template>
  <div>
    <h1 class="text-2xl font-bold text-gray-900 font-serif">Blog</h1>
    <div class="mt-6 space-y-0">
      <router-link
        v-for="post in sortedPosts"
        :key="post.id"
        :to="`/blog/${post.id}`"
        class="flex items-baseline gap-4 py-4 border-b border-gray-100 last:border-0 group"
      >
        <span class="text-xs text-gray-400 whitespace-nowrap font-mono tabular-nums">{{ formatDate(post.date) }}</span>
        <span class="text-gray-800 group-hover:text-gray-500 transition-colors text-base font-serif">
          {{ post.title }}
        </span>
      </router-link>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { posts } from '@/data/posts'

const sortedPosts = computed(() =>
  [...posts].sort((a, b) => new Date(b.date) - new Date(a.date))
)

function formatDate(dateStr) {
  const d = new Date(dateStr)
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  return `${months[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`
}
</script>
