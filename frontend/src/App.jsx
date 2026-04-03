import { useState, useRef, useEffect, useCallback } from 'react';
import ModDashboard from './dashboard/ModDashboard';
import {
  Menu,
  User,
  Send,
  ThumbsUp,
  ThumbsDown,
  ExternalLink,
  Info,
  Plus,
  Settings,
  MoreVertical,
  Trash2,
  Loader2,
} from "lucide-react";
import gcLogo from './images/logo.png';
import translations from './translations';

// --- Components ---

const GCHeader = ({ isLoggedIn, onLogout, language, onLanguageChange }) => {
  const t = translations[language];
  return (
    <header className="bg-white border-b border-gc-border py-4 px-6 md:px-12 flex items-center justify-between shrink-0 relative z-20">
      <div className="flex items-center gap-4">
        <div className="h-8 flex items-center gap-1">
          <img src={gcLogo} alt="" className="h-[140px] w-auto" />
        </div>
      </div>

      <div className="flex items-center gap-6 text-sm">
        <div className="flex gap-1">
          <button
            onClick={() => onLanguageChange("en")}
            className={`px-3 py-1 rounded border text-sm font-bold transition-colors ${language === "en" ? "bg-[#26374a] text-white border-[#26374a]" : "bg-white text-[#26374a] border-gray-300 hover:bg-gray-50"}`}
          >
            EN
          </button>
          <button
            onClick={() => onLanguageChange("fr")}
            className={`px-3 py-1 rounded border text-sm font-bold transition-colors ${language === "fr" ? "bg-[#26374a] text-white border-[#26374a]" : "bg-white text-[#26374a] border-gray-300 hover:bg-gray-50"}`}
          >
            FR
          </button>
        </div>
        {isLoggedIn && (
          <button
            onClick={onLogout}
            className="flex items-center gap-2 bg-gray-100 hover:bg-gray-200 px-4 py-2 rounded border border-gray-300 transition-colors"
          >
            <User size={16} className="text-gc-dark" />
            <span className="font-medium text-gc-dark">{t.signOut}</span>
          </button>
        )}
      </div>
    </header>
  );
};

const GCLogin = ({ onLogin, language, onLanguageChange }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(false);
  const t = translations[language];

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: email, password })
      });
      const data = await res.json();
      if (res.ok) {
        onLogin(data);
      } else {
        setError(true);
      }
    } catch {
      setError(true);
    }
  };

  return (
    <div className="min-h-screen bg-[#F5F5F5] flex flex-col">
      <GCHeader isLoggedIn={false} language={language} onLanguageChange={onLanguageChange} />

      <div className="flex-1 flex items-center justify-center p-4">
        <div className="bg-white p-8 md:p-12 shadow-sm border border-gray-200 w-full max-w-[500px]">
          <h2 className="text-3xl font-bold text-gray-800 mb-8">{t.signIn}</h2>

          {error && (
            <div className="bg-[#F3E9E8] border-l-4 border-gc-red p-4 mb-6 flex items-start gap-3">
              <div className="bg-gc-red rounded-full p-0.5 mt-0.5 text-white flex items-center justify-center w-5 h-5 font-bold text-xs">!</div>
              <p className="text-gray-800 text-sm font-medium">
                {t.loginError}
              </p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-bold text-gray-700 mb-2">
                {t.emailLabel}
              </label>
              <input
                type="text"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full border border-gray-400 p-2 rounded-sm focus:ring-2 focus:ring-gc-blue focus:border-gc-blue outline-none"
                placeholder={t.emailPlaceholder}
              />
            </div>

            <div>
              <label className="block text-sm font-bold text-gray-700 mb-2">
                {t.passwordLabel}
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full border border-gray-400 p-2 rounded-sm focus:ring-2 focus:ring-gc-blue focus:border-gc-blue outline-none"
                placeholder={t.passwordPlaceholder}
              />
            </div>

            <button
              type="submit"
              className="w-40 bg-gc-blue text-white font-medium py-3 px-4 rounded-sm hover:bg-[#1b2a3a] transition-colors"
            >
              {t.signIn}
            </button>
          </form>

          <div className="mt-8 text-center">
            <a href="#" className="text-gc-link underline text-sm hover:text-gc-blue">{t.forgotPassword}</a>
          </div>

          <hr className="my-8 border-gray-200" />

          <p className="text-xs text-gray-500 text-center">
            {t.authorizedOnly}
          </p>
        </div>
      </div>

      <footer className="py-6 px-12 bg-white border-t border-gray-200 flex gap-6 text-sm text-gc-link">
        <a href="#">{t.termsAndConditions}</a>
        <a href="#">{t.privacy}</a>
      </footer>
    </div>
  );
};

function getMessageContext(messages, messageId) {
  const botMsgIndex = messages.findIndex((m) => m.id === messageId);
  const botMsg = botMsgIndex >= 0 ? messages[botMsgIndex] : null;
  const userMsg = botMsgIndex > 0 ? messages[botMsgIndex - 1] : { text: "" };
  return { botMsgIndex, botMsg, userMsg };
}

function uniqueChunkIds(citations) {
  if (!Array.isArray(citations)) return [];
  const seen = new Set();
  const ids = [];
  for (const citation of citations) {
    const chunkId = citation?.chunk_id;
    if (!chunkId || seen.has(chunkId)) continue;
    seen.add(chunkId);
    ids.push(chunkId);
  }
  return ids;
}

function defaultAnswerFeedbackState() {
  return {
    thumb: "none",
    persistedThumb: "none",
    comment: "",
    showComposer: false,
  };
}

const ChatInterface = ({ user, onLogout, language, onLanguageChange }) => {
  const t = translations[language];
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const scrollRef = useRef(null);
  const [reviewPerCitation, setReviewPerCitation] = useState({});
  const [answerFeedback, setAnswerFeedback] = useState({});

  const [conversations, setConversations] = useState([]);
  const [activeConversation, setActiveConversation] = useState(null);
  const [openMenuId, setOpenMenuId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const handleClickOutside = () => setOpenMenuId(null);
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  const fetchHistory = useCallback(async () => {
    try {
      const res = await fetch(`/api/conversations?user_id=${user.user_id}`);
      const data = await res.json();
      setConversations(data);
    } catch (error) {
      console.error("Failed to fetch history:", error);
    }
  }, [user.user_id]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const loadConversation = async (convId) => {
    setActiveConversation(convId);
    try {
      const res = await fetch(`/api/conversations/${convId}/messages`);
      const data = await res.json();
      const formattedData = data.map((msg, i) => ({ ...msg, id: Date.now() + i }));
      setMessages(formattedData);
      if (data.length > 0 && data[data.length - 1].language) {
        onLanguageChange(data[data.length - 1].language);
      }
      if (window.innerWidth < 768) setSidebarOpen(false);
    } catch (error) {
      console.error("Failed to load conversation:", error);
    }
  };

  const handleNewChat = () => {
    setActiveConversation(null);
    setMessages([]);
    setReviewPerCitation({});
    setAnswerFeedback({});
    if (window.innerWidth < 768) setSidebarOpen(false);
  };

  const handleDeleteChat = async (e, convId) => {
    e.stopPropagation();
    try {
      await fetch(`/api/conversations/${convId}`, { method: 'DELETE' });
      setConversations((prev) => prev.filter((c) => c.id !== convId));
      if (activeConversation === convId) handleNewChat();
      setOpenMenuId(null);
    } catch (error) {
      console.error("Failed to delete conversation:", error);
    }
  };

  const toggleMenu = (e, convId) => {
    e.stopPropagation();
    setOpenMenuId(openMenuId === convId ? null : convId);
  };

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleLanguageChange = async (newLang) => {
    onLanguageChange(newLang);
    try {
      await fetch('/api/language', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: user.user_id, language: newLang })
      });
    } catch (error) {
      console.error("Error saving language preference:", error);
    }
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = { id: Date.now(), type: "user", text: input, language };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: input,
          language,
          user_id: user.user_id,
          conversation_id: activeConversation,
        }),
      });
      const data = await response.json();

      if (!activeConversation && data.conversation_id) {
        setActiveConversation(data.conversation_id);
        fetchHistory();
      }

      const botMsg = {
        id: Date.now() + 1,
        type: "bot",
        text: data.reply,
        citations: data.citations || null,
        language,
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setIsLoading(false); // 3. Stop loading
    }
  };

  const submitFeedback = async ({
    thumb,
    turnId,
    question,
    answer,
    comment = "",
    citedChunkIds = [],
    feedbackType,
  }) => {
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        thumb,
        conversation_id: activeConversation ? activeConversation.toString() : "default",
        turn_id: turnId,
        question,
        answer,
        comment,
        cited_chunk_ids: citedChunkIds,
        feedback_type: feedbackType,
      }),
    });
  };

  const handleCitationFeedback = async (messageId, citationIndex, action) => {
    const currentAction = reviewPerCitation[messageId]?.[citationIndex];
    const newAction = currentAction === action ? 'none' : action;

    setReviewPerCitation((prev) => ({
      ...prev,
      [messageId]: {
        ...prev[messageId],
        [citationIndex]: newAction,
      },
    }));

    const { botMsg, userMsg } = getMessageContext(messages, messageId);
    const clickedCitation = botMsg?.citations?.[citationIndex];
    const chunkIdsToSend = clickedCitation?.chunk_id ? [clickedCitation.chunk_id] : [];

    try {
      await submitFeedback({
        thumb: newAction,
        turnId: messageId.toString(),
        question: userMsg.text,
        answer: botMsg?.text || "",
        citedChunkIds: chunkIdsToSend,
        feedbackType: "citation",
      });
    } catch (error) {
      console.error("Error submitting feedback:", error);
    }
  };

  const handleAnswerThumb = async (messageId, action) => {
    const current = answerFeedback[messageId] || defaultAnswerFeedbackState();
    const { botMsg, userMsg } = getMessageContext(messages, messageId);
    const allChunkIds = uniqueChunkIds(botMsg?.citations);

    if (action === "up") {
      if (current.showComposer && current.persistedThumb === "up") {
        setAnswerFeedback((prev) => ({
          ...prev,
          [messageId]: {
            ...current,
            thumb: "up",
            showComposer: false,
            comment: "",
          },
        }));
        return;
      }

      const nextThumb = current.persistedThumb === "up" ? "none" : "up";
      setAnswerFeedback((prev) => ({
        ...prev,
        [messageId]: {
          ...current,
          thumb: nextThumb,
          persistedThumb: nextThumb,
          showComposer: false,
          comment: "",
        },
      }));

      try {
        await submitFeedback({
          thumb: nextThumb,
          turnId: messageId.toString(),
          question: userMsg.text,
          answer: botMsg?.text || "",
          citedChunkIds: allChunkIds,
          feedbackType: "answer",
        });
      } catch (error) {
        console.error("Error submitting answer feedback:", error);
      }
      return;
    }

    if (current.showComposer) {
      setAnswerFeedback((prev) => ({
        ...prev,
        [messageId]: {
          ...current,
          thumb: current.persistedThumb,
          showComposer: false,
          comment: "",
        },
      }));
      return;
    }

    if (current.persistedThumb === "down") {
      setAnswerFeedback((prev) => ({
        ...prev,
        [messageId]: defaultAnswerFeedbackState(),
      }));

      try {
        await submitFeedback({
          thumb: "none",
          turnId: messageId.toString(),
          question: userMsg.text,
          answer: botMsg?.text || "",
          citedChunkIds: allChunkIds,
          feedbackType: "answer",
        });
      } catch (error) {
        console.error("Error clearing answer feedback:", error);
      }
      return;
    }

    setAnswerFeedback((prev) => ({
      ...prev,
      [messageId]: {
        ...current,
        thumb: "down",
        showComposer: true,
      },
    }));
  };

  const handleAnswerCommentChange = (messageId, value) => {
    const current = answerFeedback[messageId] || defaultAnswerFeedbackState();
    setAnswerFeedback((prev) => ({
      ...prev,
      [messageId]: {
        ...current,
        thumb: "down",
        showComposer: true,
        comment: value,
      },
    }));
  };

  const finalizeAnswerDownvote = async (messageId, comment = "") => {
    const current = answerFeedback[messageId] || defaultAnswerFeedbackState();
    const { botMsg, userMsg } = getMessageContext(messages, messageId);
    const allChunkIds = uniqueChunkIds(botMsg?.citations);

    setAnswerFeedback((prev) => ({
      ...prev,
      [messageId]: {
        ...current,
        thumb: "down",
        persistedThumb: "down",
        showComposer: false,
        comment,
      },
    }));

    try {
      await submitFeedback({
        thumb: "down",
        turnId: messageId.toString(),
        question: userMsg.text,
        answer: botMsg?.text || "",
        comment,
        citedChunkIds: allChunkIds,
        feedbackType: "answer",
      });
    } catch (error) {
      console.error("Error submitting answer feedback:", error);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-white">
      <div className="border-b border-gray-300">
        <GCHeader
          isLoggedIn={true}
          onLogout={onLogout}
          language={language}
          onLanguageChange={handleLanguageChange}
        />
        <div className="px-6 md:px-12 py-2 text-sm text-gray-600 bg-white border-b border-gray-200">
          <span className="underline cursor-pointer">{t.breadcrumbHome}</span>
          <span className="mx-2 text-gray-400">&gt;</span>
          <span className="underline cursor-pointer">{t.breadcrumbChat}</span>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside
          className={`${isSidebarOpen ? "w-72" : "w-0"} bg-white border-r border-gray-200 flex flex-col transition-all duration-300 overflow-hidden`}
        >
          <div className="p-4">
            <button
              onClick={handleNewChat}
              className="w-full bg-gc-blue text-white py-3 px-4 rounded flex items-center justify-center gap-2 font-medium hover:bg-[#1b2a3a]"
            >
              <Plus size={18} />
              {t.newChat}
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-2">
            <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wide px-3 mb-2 mt-2">
              {t.history}
            </h3>
            <ul className="space-y-1">
              {conversations.map((conv) => (
                <li
                  key={conv.id}
                  onClick={() => loadConversation(conv.id)}
                  className={`px-3 py-2 text-sm cursor-pointer border-l-4 group relative flex justify-between items-center ${activeConversation === conv.id ? "bg-[#E6EEF5] border-gc-blue font-medium text-gray-800" : "border-transparent text-gray-600 hover:bg-gray-100"}`}
                >
                  <span className="truncate pr-2">{conv.title}</span>
                  <button
                    onClick={(e) => toggleMenu(e, conv.id)}
                    className="p-1 rounded hover:bg-gray-300 text-gray-400 hover:text-gray-700 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <MoreVertical size={14} />
                  </button>
                  {openMenuId === conv.id && (
                    <div className="absolute right-2 top-8 w-28 bg-white border border-gray-200 shadow-md rounded-md z-50 overflow-hidden">
                      <button
                        onClick={(e) => handleDeleteChat(e, conv.id)}
                        className="w-full text-left px-3 py-2 text-xs text-red-600 hover:bg-red-50 flex items-center gap-2"
                      >
                        <Trash2 size={12} /> {t.delete}
                      </button>
                    </div>
                  )}
                </li>
              ))}
            </ul>
          </div>
        </aside>

        {/* Main Chat Area */}
        <main className="flex-1 flex flex-col bg-white relative">
          {/* Mobile Toggle */}
          <button
            onClick={() => setSidebarOpen(!isSidebarOpen)}
            className="absolute top-4 left-4 p-2 bg-gray-100 rounded md:hidden z-10"
          >
            <Menu size={20} />
          </button>

          <div className="flex-1 overflow-y-auto p-6 md:p-12 space-y-8">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-gray-400">
                <p>{t.startConversation}</p>
              </div>
            )}

            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex flex-col ${msg.type === "user" ? "items-end" : "items-start"}`}
              >
                {msg.type === "user" ? (
                  <>
                    <span className="text-xs text-gray-500 mb-1 mr-1">
                      {t.citizen}
                    </span>
                    <div className="bg-[#DEE8F4] text-gray-800 p-5 rounded-lg max-w-2xl text-sm leading-relaxed">
                      {msg.text}
                    </div>
                  </>
                ) : (
                  <div className="max-w-3xl">
                    {(() => {
                      const answerState =
                        answerFeedback[msg.id] || defaultAnswerFeedbackState();
                      const isAnswerUp = answerState.thumb === "up";
                      const isAnswerDown = answerState.thumb === "down";
                      const showAnswerFeedback =
                        messages.findIndex((item) => item.id === msg.id) > 0;

                      return (
                        <>
                          <div className="flex items-center gap-2 mb-2">
                            {/* Red Bot Icon */}
                            <div className="w-5 h-5 rounded-full bg-gc-red flex items-center justify-center text-white">
                              <span className="text-[10px] font-bold">Bot</span>
                            </div>
                            <span className="text-sm font-bold text-gray-700">
                              {t.botName}
                            </span>
                          </div>

                          <div className="bg-white border border-gray-200 p-6 rounded-lg text-sm leading-relaxed text-gray-800 shadow-sm">
                            <p className="whitespace-pre-wrap">{msg.text}</p>
                          </div>

                          {showAnswerFeedback && (
                            <div className="mt-3 border border-gray-200 rounded p-3 bg-gray-50 space-y-3">
                              <div className="flex items-center justify-between gap-3 flex-wrap">
                                <div>
                                  <p className="text-[10px] font-bold text-gray-500 uppercase">
                                    {t.answerFeedback}
                                  </p>
                                  <p className="text-xs text-gray-600">
                                    {t.answerFeedbackHint}
                                  </p>
                                </div>
                                <div className="flex items-center gap-2">
                                  <button
                                    onClick={() =>
                                      handleAnswerThumb(msg.id, "up")
                                    }
                                    className={`p-1 rounded ${isAnswerUp ? "text-gc-blue" : "text-gray-400 hover:text-gc-blue"}`}
                                    aria-pressed={isAnswerUp}
                                  >
                                    <ThumbsUp size={16} />
                                  </button>
                                  <button
                                    onClick={() =>
                                      handleAnswerThumb(msg.id, "down")
                                    }
                                    className={`p-1 rounded ${isAnswerDown ? "text-gc-red" : "text-gray-400 hover:text-gc-red"}`}
                                    aria-pressed={isAnswerDown}
                                  >
                                    <ThumbsDown size={16} />
                                  </button>
                                </div>
                              </div>

                              {answerState.showComposer && (
                                <div className="space-y-2">
                                  <label className="block text-xs font-medium text-gray-700">
                                    {t.downvotePrompt}
                                  </label>
                                  <textarea
                                    value={answerState.comment}
                                    onChange={(e) =>
                                      handleAnswerCommentChange(
                                        msg.id,
                                        e.target.value,
                                      )
                                    }
                                    rows={3}
                                    className="w-full border border-gray-300 rounded px-3 py-2 text-sm text-gray-800 focus:ring-2 focus:ring-gc-blue focus:border-gc-blue outline-none"
                                    placeholder={t.optionalComment}
                                  />
                                  <div className="flex items-center gap-2">
                                    <button
                                      type="button"
                                      className="px-3 py-1.5 rounded bg-gc-blue text-white text-sm hover:bg-[#1b2a3a]"
                                      onClick={() =>
                                        finalizeAnswerDownvote(
                                          msg.id,
                                          answerState.comment.trim(),
                                        )
                                      }
                                    >
                                      {t.submit}
                                    </button>
                                    <button
                                      type="button"
                                      className="px-3 py-1.5 rounded border border-gray-300 text-sm text-gray-700 hover:bg-gray-100"
                                      onClick={() =>
                                        finalizeAnswerDownvote(msg.id, "")
                                      }
                                    >
                                      {t.skip}
                                    </button>
                                  </div>
                                </div>
                              )}
                            </div>
                          )}

                          {/* Citation Block matching image_5f35b3 */}
                          {msg.citations && msg.citations.length > 0 && (
                            <div className="mt-3 space-y-2">
                              {msg.citations.map((citation, index) => {
                                const isUp =
                                  reviewPerCitation[msg.id]?.[index] === "up";
                                const isDown =
                                  reviewPerCitation[msg.id]?.[index] === "down";

                                return (
                                  <div
                                    key={index}
                                    className="bg-[#F8F9FA] border border-gray-200 rounded p-3 flex items-center justify-between"
                                  >
                                    <div className="flex items-center gap-3">
                                      <div className="bg-gc-blue rounded-full p-1">
                                        <Info
                                          size={12}
                                          className="text-white"
                                        />
                                      </div>
                                      <div className="flex flex-col">
                                        <span className="text-[10px] font-bold text-gray-500 uppercase">
                                          {t.officialSource}
                                        </span>
                                        <a
                                          href={citation.link}
                                          className="text-xs text-gc-link font-medium hover:underline flex items-center gap-1"
                                        >
                                          {citation.title}
                                          <ExternalLink size={10} />
                                        </a>
                                      </div>
                                    </div>

                                    <div className="flex items-center gap-2">
                                      <button
                                        onClick={() =>
                                          handleCitationFeedback(
                                            msg.id,
                                            index,
                                            "up",
                                          )
                                        }
                                        className={`p-1 rounded ${isUp ? "text-gc-blue" : "text-gray-400 hover:text-gc-blue"}`}
                                        aria-pressed={isUp}
                                      >
                                        <ThumbsUp size={16} />
                                      </button>

                                      <button
                                        onClick={() =>
                                          handleCitationFeedback(
                                            msg.id,
                                            index,
                                            "down",
                                          )
                                        }
                                        className={`p-1 rounded ${isDown ? "text-gc-red" : "text-gray-400 hover:text-gc-red"}`}
                                        aria-pressed={isDown}
                                      >
                                        <ThumbsDown size={16} />
                                      </button>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </>
                      );
                    })()}
                  </div>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="flex flex-col items-start animate-in fade-in slide-in-from-bottom-2 duration-300">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-5 h-5 rounded-full bg-gc-red flex items-center justify-center text-white">
                    <span className="text-[10px] font-bold">Bot</span>
                  </div>
                  <span className="text-sm font-bold text-gray-700">
                    {t.botName}
                  </span>
                </div>
                <div className="bg-white border border-gray-200 p-4 rounded-lg shadow-sm flex items-center gap-3">
                  <Loader2 className="h-4 w-4 animate-spin text-gc-blue" />
                  <span className="text-sm text-gray-500 italic">
                    {t.botIsThinking || "Thinking..."}
                  </span>
                </div>
              </div>
            )}
            <div ref={scrollRef} />
          </div>

          <div className="p-6 md:px-12 border-t border-gray-200 bg-white">
            <form onSubmit={handleSend} className="max-w-4xl mx-auto relative">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                className="w-full border border-gray-400 p-4 pr-12 rounded-sm shadow-inner focus:ring-2 focus:ring-gc-blue focus:border-gc-blue outline-none text-sm"
                placeholder={t.inputPlaceholder}
              />
              <button
                type="submit"
                className="absolute right-3 top-1/2 -translate-y-1/2 bg-gc-blue text-white p-1.5 rounded-sm hover:bg-[#1b2a3a]"
              >
                <Send size={18} />
              </button>
            </form>
            <div className="flex justify-between items-center max-w-4xl mx-auto mt-2 px-1">
              <p className="text-xs text-gray-500">{t.aiDisclaimer}</p>
              <div className="flex gap-4 text-xs text-gc-link">
                <a href="#">{t.termsAndConditions}</a>
                <a href="#">{t.privacyPolicy}</a>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

// --- Main App Shell ---

function App() {
  const [user, setUser] = useState(null);
  const [language, setLanguage] = useState("en");

  if (window.location.pathname === '/mod-dashboard') {
    return <ModDashboard />;
  }

  const handleLogin = (userData) => {
    setUser(userData);
    if (userData.language) setLanguage(userData.language);
  };

  return (
    <>
      {user ? (
        <ChatInterface user={user} onLogout={() => setUser(null)} language={language} onLanguageChange={setLanguage} />
      ) : (
        <GCLogin onLogin={handleLogin} language={language} onLanguageChange={setLanguage} />
      )}
    </>
  );
}

export default App;
