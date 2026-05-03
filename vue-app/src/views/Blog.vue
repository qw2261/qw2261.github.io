<template>
  <div>
    <h1 class="text-2xl font-bold text-gray-900">Blog</h1>
    <div class="mt-4 space-y-3">
      <div
        v-for="post in sortedPosts"
        :key="post.id"
        class="flex items-baseline gap-3 py-2 border-b border-gray-100 last:border-0"
      >
        <span class="text-sm text-gray-400 whitespace-nowrap">{{ formatDate(post.date) }}</span>
        <router-link
          :to="`/blog/${post.id}`"
          class="text-blue-600 hover:underline text-lg"
        >
          {{ post.title }}
        </router-link>
      </div>
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
