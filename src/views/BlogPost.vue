<template>
  <div>
    <div v-if="post">
      <h1 class="text-2xl font-bold text-gray-900 font-serif">{{ post.title }}</h1>
      <p class="text-xs text-gray-400 mt-2 font-mono">{{ formattedDate }}</p>
      <div
        class="mt-6 prose max-w-none text-gray-800 leading-relaxed space-y-4 font-serif"
        v-html="post.content"
      ></div>
      <div class="mt-8 pt-4 border-t border-gray-100">
        <router-link to="/blog" class="text-sm text-gray-500 hover:text-gray-800 transition-colors">
          ← Back to Blog
        </router-link>
      </div>
    </div>
    <div v-else class="text-center py-12">
      <p class="text-gray-500">Post not found.</p>
      <router-link to="/blog" class="text-sm text-gray-500 hover:text-gray-800 mt-2 inline-block transition-colors">
        ← Back to Blog
      </router-link>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { getPostById } from '@/data/posts'

const route = useRoute()
const post = ref(null)

onMounted(async () => {
  post.value = await getPostById(route.params.id)
})

const formattedDate = computed(() => {
  if (!post.value) return ''
  const d = new Date(post.value.date)
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
})
</script>