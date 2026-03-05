import { useState, useRef, useEffect } from 'react';
import ModDashboard from './dashboard/ModDashboard';
import {
  Menu, User, Send, ThumbsUp, ThumbsDown,
  ExternalLink, Info, Plus, Settings, LogOut
} from 'lucide-react';
import gcLogo from './images/logo.png';

// --- Components ---

const GCHeader = ({ isLoggedIn, onLogout }) => (
  <header className="bg-white border-b border-gc-border py-4 px-6 md:px-12 flex items-center justify-between shrink-0 relative z-20">
    {/* Left: Branding */}
    <div className="flex items-center gap-4">
      {/* Flag Logo (CSS construction or SVG placeholder) */}
      <div className="h-8 flex items-center gap-1">
        <img
          src={gcLogo}
          alt=""
          className="h-[140px] w-auto"
        />
      </div>
    </div>

    {/* Right: Language & Auth */}
    <div className="flex items-center gap-6 text-sm">
      <a href="#" className="underline text-gc-link hover:text-gc-blue">Français</a>
      {isLoggedIn && (
        <button
          onClick={onLogout}
          className="flex items-center gap-2 bg-gray-100 hover:bg-gray-200 px-4 py-2 rounded border border-gray-300 transition-colors"
        >
          <User size={16} className="text-gc-dark" />
          <span className="font-medium text-gc-dark">Sign Out</span>
        </button>
      )}
    </div>
  </header>
);

const GCLogin = ({ onLogin }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (email && password) {
      onLogin();
    } else {
      setError(true);
    }
  };

  return (
    <div className="min-h-screen bg-[#F5F5F5] flex flex-col">
      <GCHeader isLoggedIn={false} />

      <div className="flex-1 flex items-center justify-center p-4">
        <div className="bg-white p-8 md:p-12 shadow-sm border border-gray-200 w-full max-w-[500px]">
          <h2 className="text-3xl font-bold text-gray-800 mb-8">Sign in</h2>

          {/* Error Banner matching image_5f32ec */}
          {error && (
            <div className="bg-[#F3E9E8] border-l-4 border-gc-red p-4 mb-6 flex items-start gap-3">
              <div className="bg-gc-red rounded-full p-0.5 mt-0.5 text-white flex items-center justify-center w-5 h-5 font-bold text-xs">!</div>
              <p className="text-gray-800 text-sm font-medium">
                The employee ID, email or password you entered is incorrect. Please try again.
              </p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-bold text-gray-700 mb-2">
                Employee ID or Email
              </label>
              <input
                type="text"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full border border-gray-400 p-2 rounded-sm focus:ring-2 focus:ring-gc-blue focus:border-gc-blue outline-none"
                placeholder="username"
              />
            </div>

            <div>
              <label className="block text-sm font-bold text-gray-700 mb-2">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full border border-gray-400 p-2 rounded-sm focus:ring-2 focus:ring-gc-blue focus:border-gc-blue outline-none"
                placeholder="********"
              />
            </div>

            <button
              type="submit"
              className="w-40 bg-gc-blue text-white font-medium py-3 px-4 rounded-sm hover:bg-[#1b2a3a] transition-colors"
            >
              Sign in
            </button>
          </form>

          <div className="mt-8 text-center">
            <a href="#" className="text-gc-link underline text-sm hover:text-gc-blue">Forgot your password?</a>
          </div>

          <hr className="my-8 border-gray-200" />

          <p className="text-xs text-gray-500 text-center">
            This system is for authorized Government of Canada personnel only.
          </p>
        </div>
      </div>

      <footer className="py-6 px-12 bg-white border-t border-gray-200 flex gap-6 text-sm text-gc-link">
        <a href="#">Terms and conditions</a>
        <a href="#">Privacy</a>
      </footer>
    </div>
  );
};

const ChatInterface = ({ onLogout }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      text: "Hello. I am PolicyAI. I can help you navigate Government of Canada regulations, finding information on taxes, immigration, environment, and more.\nHow can I assist you today?",
      citations: null
    }
  ]);
  const [input, setInput] = useState('');
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const scrollRef = useRef(null);
  const [reviewPerCitation, setReviewPerCitation] = useState({});

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = { id: Date.now(), type: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });
      const data = await response.json();

      const botMsg = {
        id: Date.now() + 1,
        type: "bot",
        text: data.reply,
        citations: data.citations || null,
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (error) {
      console.error("Error:", error);
    }
  };


  const handleThumbsUpDown = async (messageId, citationIndex, action) => {
    // 1. Determine the new toggle state ('up', 'down', or 'none')
    const currentAction = reviewPerCitation[messageId]?.[citationIndex];
    const newAction = currentAction === action ? 'none' : action;

    // 2. Update the UI instantly
    setReviewPerCitation((prev) => ({
      ...prev,
      [messageId]: {
        ...prev[messageId],
        [citationIndex]: newAction,
      },
    }));

    // 3. Find the associated question, answer, and specifically the chunk ID they clicked
    const botMsgIndex = messages.findIndex((m) => m.id === messageId);
    const botMsg = messages[botMsgIndex];
    const userMsg = botMsgIndex > 0 ? messages[botMsgIndex - 1] : { text: "" };
    
    // Grab the specific citation they voted on
    const clickedCitation = botMsg.citations[citationIndex];
    // Put it in an array because the backend expects a list of IDs
    const chunkIdsToSend = clickedCitation.chunk_id ? [clickedCitation.chunk_id] : [];

    // 4. Fire the API call with the chunk_id included
    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          thumb: newAction,
          conversation_id: "default", 
          turn_id: messageId.toString(),
          question: userMsg.text,
          answer: botMsg.text,
          cited_chunk_ids: chunkIdsToSend, // <--- NOW INCLUDED HERE
        }),
      });
    } catch (error) {
      console.error("Error submitting feedback:", error);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-white">
      {/* Top Header */}
      <div className="border-b border-gray-300">
        <GCHeader isLoggedIn={true} onLogout={onLogout} />
        {/* Breadcrumbs */}
        <div className="px-6 md:px-12 py-2 text-sm text-gray-600 bg-white border-b border-gray-200">
          <span className="underline cursor-pointer">Canada.ca</span>
          <span className="mx-2 text-gray-400">&gt;</span>
          <span className="underline cursor-pointer">ChatBot</span>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className={`${isSidebarOpen ? 'w-72' : 'w-0'} bg-white border-r border-gray-200 flex flex-col transition-all duration-300 overflow-hidden`}>
          <div className="p-4">
            <button className="w-full bg-gc-blue text-white py-3 px-4 rounded flex items-center justify-center gap-2 font-medium hover:bg-[#1b2a3a]">
              <Plus size={18} />
              New Chat
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-2">
            <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wide px-3 mb-2 mt-2">History</h3>
            <ul className="space-y-1">
              <li className="bg-[#E6EEF5] border-l-4 border-gc-blue px-3 py-2 text-sm font-medium text-gray-800 cursor-pointer">
                Carbon Tax Inquiry
              </li>
              <li className="px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 cursor-pointer border-l-4 border-transparent">
                Small Business Tax
              </li>
              <li className="px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 cursor-pointer border-l-4 border-transparent">
                EI Eligibility
              </li>
            </ul>

            <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wide px-3 mb-2 mt-8">Topics</h3>
            <ul className="space-y-2 px-3 text-sm text-gc-link">
              <li className="cursor-pointer hover:underline">Immigration</li>
              <li className="cursor-pointer hover:underline">Taxation</li>
              <li className="cursor-pointer hover:underline">Environment</li>
              <li className="cursor-pointer hover:underline">Health</li>
            </ul>
          </div>

          <div className="p-4 border-t border-gray-200">
            <button className="flex items-center gap-2 text-sm text-gray-600 hover:text-gc-blue w-full p-2 rounded hover:bg-gray-50">
              <Settings size={16} />
              Settings
            </button>
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
            {messages.map((msg) => (
              <div key={msg.id} className={`flex flex-col ${msg.type === 'user' ? 'items-end' : 'items-start'}`}>

                {msg.type === 'user' ? (
                  <>
                    <span className="text-xs text-gray-500 mb-1 mr-1">Citizen</span>
                    <div className="bg-[#DEE8F4] text-gray-800 p-5 rounded-lg max-w-2xl text-sm leading-relaxed">
                      {msg.text}
                    </div>
                  </>
                ) : (
                  <div className="max-w-3xl">
                    <div className="flex items-center gap-2 mb-2">
                      {/* Red Bot Icon */}
                      <div className="w-5 h-5 rounded-full bg-gc-red flex items-center justify-center text-white">
                        <span className="text-[10px] font-bold">Bot</span>
                      </div>
                      <span className="text-sm font-bold text-gray-700">PolicyAI</span>
                    </div>

                    <div className="bg-white border border-gray-200 p-6 rounded-lg text-sm leading-relaxed text-gray-800 shadow-sm">
                      <p className="whitespace-pre-wrap">{msg.text}</p>
                    </div>

                    {/* Citation Block matching image_5f35b3 */}
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="mt-3 space-y-2">
                        {msg.citations.map((citation, index) => {
                          const isUp = reviewPerCitation[msg.id]?.[index] === 'up';
                          const isDown = reviewPerCitation[msg.id]?.[index] === 'down';

                          return (
                            <div
                              key={index}
                              className="bg-[#F8F9FA] border border-gray-200 rounded p-3 flex items-center justify-between"
                            >
                              <div className="flex items-center gap-3">
                                <div className="bg-gc-blue rounded-full p-1">
                                  <Info size={12} className="text-white" />
                                </div>
                                <div className="flex flex-col">
                                  <span className="text-[10px] font-bold text-gray-500 uppercase">
                                    Official Source
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
                                  onClick={() => handleThumbsUpDown(msg.id, index, 'up')}
                                  className={`p-1 rounded ${isUp ? 'text-gc-blue' : 'text-gray-400 hover:text-gc-blue'}`}
                                  aria-pressed={isUp}
                                >
                                  <ThumbsUp size={16} />
                                </button>

                                <button
                                  onClick={() => handleThumbsUpDown(msg.id, index, 'down')}
                                  className={`p-1 rounded ${isDown ? 'text-gc-red' : 'text-gray-400 hover:text-gc-red'}`}
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
                  </div>
                )}
              </div>
            ))}
            <div ref={scrollRef} />
          </div>

          {/* Input Area matching image_5f35b3 footer */}
          <div className="p-6 md:px-12 border-t border-gray-200 bg-white">
            <form onSubmit={handleSend} className="max-w-4xl mx-auto relative">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                className="w-full border border-gray-400 p-4 pr-12 rounded-sm shadow-inner focus:ring-2 focus:ring-gc-blue focus:border-gc-blue outline-none text-sm"
                placeholder="Type your policy question here..."
              />
              <button
                type="submit"
                className="absolute right-3 top-1/2 -translate-y-1/2 bg-gc-blue text-white p-1.5 rounded-sm hover:bg-[#1b2a3a]"
              >
                <Send size={18} />
              </button>
            </form>
            <div className="flex justify-between items-center max-w-4xl mx-auto mt-2 px-1">
              <p className="text-xs text-gray-500">
                AI responses are for informational purposes. Verify with official sources.
              </p>
              <div className="flex gap-4 text-xs text-gc-link">
                <a href="#">Terms and conditions</a>
                <a href="#">Privacy Policy</a>
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
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // Hidden moderator dashboard, accessible only via direct URL.
  if (window.location.pathname === '/mod-dashboard') {
    return <ModDashboard />;
  }

  return (
    <>
      {isLoggedIn ? (
        <ChatInterface onLogout={() => setIsLoggedIn(false)} />
      ) : (
        <GCLogin onLogin={() => setIsLoggedIn(true)} />
      )}
    </>
  );
}

export default App;