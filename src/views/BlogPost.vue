<template>
  <div>
    <div v-if="post">
      <h1 class="text-2xl font-bold text-gray-900 font-serif">{{ post.title }}</h1>
      <p class="text-xs text-gray-400 mt-2 font-mono">{{ formattedDate }}</p>
      <div
        class="mt-6 prose max-w-none text-gray-800 leading-relaxed space-y-4 font-serif"
        v-html="post.content"
      ></div>

      <div class="mt-8 flex items-center gap-5 py-3 border-t border-b border-gray-100">
        <button
          @click="sharePost"
          class="flex items-center gap-1.5 text-sm text-gray-400 hover:text-blue-500 transition-all duration-200 hover:scale-110 active:scale-95"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
          </svg>
          <span>Share</span>
        </button>
        <span v-if="shareMsg" class="text-xs text-green-500 animate-pulse">{{ shareMsg }}</span>
      </div>

      <GiscusComment :term="post.title" />

      <div class="mt-6 pt-4 border-t border-gray-100">
        <router-link to="/blog" class="text-sm text-gray-500 hover:text-gray-800 transition-colors">
          ← Back to Blog
        </router-link>
      </div>
    </div>
    <div v-else-if="loading" class="text-center py-12">
      <p class="text-gray-400 text-sm">Loading...</p>
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
import GiscusComment from '@/components/GiscusComment.vue'

const route = useRoute()
const post = ref(null)
const loading = ref(true)
const shareMsg = ref('')

onMounted(async () => {
  post.value = await getPostById(route.params.id)
  loading.value = false
})

const formattedDate = computed(() => {
  if (!post.value) return ''
  const d = new Date(post.value.date)
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
})

async function sharePost() {
  const url = window.location.href
  const title = post.value?.title || 'Blog Post'

  if (navigator.share) {
    try {
      await navigator.share({ title, url })
    } catch {}
  } else {
    try {
      await navigator.clipboard.writeText(url)
      shareMsg.value = 'Link copied!'
      setTimeout(() => { shareMsg.value = '' }, 2000)
    } catch {
      shareMsg.value = 'Failed to copy'
      setTimeout(() => { shareMsg.value = '' }, 2000)
    }
  }
}
</script>