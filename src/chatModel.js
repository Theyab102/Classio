import {makeStudyId} from './studyTools';
export const newChat = () => ({id:makeStudyId(), title:'New chat', messages:[], draft:'', updatedAt:Date.now()});
export function normalizeChats(record) {
  const legacy = Array.isArray(record) ? record : record?.messages;
  let chats = Array.isArray(record?.chats) ? record.chats.filter(chat => chat && typeof chat.id === 'string' && Array.isArray(chat.messages)) : Array.isArray(legacy) && legacy.length ? [{...newChat(), title:String(legacy.find(message=>message.role==='user')?.content || 'Saved chat').slice(0,60), messages:legacy}] : [];
  if (!chats.length) chats = [newChat()];
  return {chats, activeId:chats.some(chat=>chat.id===record?.activeId)?record.activeId:chats[0].id, updatedAt:Number(record?.updatedAt)||0};
}
export function updateChat(record, id, patch) {
  return {...record, updatedAt:Date.now(), chats:record.chats.map(chat=>chat.id===id?{...chat,...patch,updatedAt:Date.now()}:chat)};
}
export function persistentChats(record) {
  return {...record, chats:record.chats.map(chat=>({...chat,messages:chat.messages.slice(-60).map(({role,content,failed})=>({role,content:String(content||'').slice(0,12000),...(failed?{failed:true}:{})}))}))};
}
