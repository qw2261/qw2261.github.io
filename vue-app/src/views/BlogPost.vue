<template>
  <div>
    <div v-if="post">
      <h1 class="text-2xl font-bold text-gray-900">{{ post.title }}</h1>
      <p class="text-sm text-gray-400 mt-1">{{ formattedDate }}</p>
      <div
        class="mt-6 prose max-w-none text-gray-800 leading-relaxed space-y-4 font-serif"
        v-html="post.content"
      ></div>
    </div>
    <div v-else class="text-center py-12">
      <p class="text-gray-500">Post not found.</p>
      <router-link to="/blog" class="text-blue-600 hover:underline mt-2 inline-block">Back to Blog</router-link>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { posts } from '@/data/posts'

const route = useRoute()
const post = computed(() => posts.find(p => p.id === route.params.id))

const formattedDate = computed(() => {
  if (!post.value) return ''
  const d = new Date(post.value.date)
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
})
</script>
