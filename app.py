import os
import base64
import secrets
import click
import re
from collections import Counter
from datetime import datetime, timedelta
from itertools import groupby
from operator import attrgetter
from threading import Thread

from flask import Flask, send_file, render_template, request, jsonify, session, url_for, redirect
from flask_migrate import Migrate
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI
from dotenv import load_dotenv
from sqlalchemy import desc
from flask_mail import Mail, Message as FlaskMessage
from flask_admin import BaseView, Admin, AdminIndexView, expose
from flask_admin.contrib.sqla import ModelView
from flask_admin.form import SecureForm
from flask.cli import with_appcontext
from pytz import timezone
import schedule
import time
from bs4 import BeautifulSoup
import requests

# Flask 애플리케이션 초기화
app = Flask(__name__)
CORS(app)  # Cross-Origin Resource Sharing 설정

# 한국 시간대 설정
KST = timezone('Asia/Seoul')

# 애플리케이션 설정
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# 환경 변수 로드 및 OpenAI 클라이언트 초기화
load_dotenv()
client = OpenAI()
migrate = Migrate(app, db)

# Flask-Mail 설정
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'mks010103@gmail.com' # 실제 이메일 주소로 변경
app.config['MAIL_PASSWORD'] = 'vhnk zrko wxxt oank' # 실제 앱 비밀번호로 변경
app.config['MAIL_DEFAULT_SENDER'] = 'mks010103@gmail.com' # 실제 이메일 주소로 변경

mail = Mail(app)

# 사용자 모델 정의
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    total_usage_time = db.Column(db.Integer, default=0)
    conversations = db.relationship('Conversation', backref='user', lazy=True)
    reset_token = db.Column(db.String(100), unique=True)
    reset_token_expiration = db.Column(db.DateTime)
    is_admin = db.Column(db.Boolean, default=False)
    # reports = db.relationship('Report', backref='user', lazy=True)
    messages = db.relationship('Message', backref='user', lazy=True)

    def set_reset_token(self):
        self.reset_token = secrets.token_urlsafe(32)
        self.reset_token_expiration = datetime.utcnow() + timedelta(hours=1)
        db.session.commit()

    def check_reset_token(self, token):
        return (self.reset_token == token and
                self.reset_token_expiration > datetime.utcnow())

# 대화 모델 정의
class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    messages = db.relationship('Message', backref='conversation', lazy=True, order_by="Message.timestamp")

# 메시지 모델 정의
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_user = db.Column(db.Boolean, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(KST))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# 관리자 뷰 보안 설정
class SecureModelView(ModelView):
    form_base_class = SecureForm
    def is_accessible(self):
        return current_user.is_authenticated and current_user.is_admin

# 관리자 인덱스 뷰 설정
class MyAdminIndexView(AdminIndexView):
    @expose('/')
    def index(self):
        if not current_user.is_authenticated or not current_user.is_admin:
            return redirect(url_for('login', next=request.url))
        return super(MyAdminIndexView, self).index()

# 관리자 페이지 설정
admin = Admin(app, name='TalKR Admin', template_mode='bootstrap3', index_view=MyAdminIndexView())
admin.add_view(SecureModelView(User, db.session))
admin.add_view(SecureModelView(Conversation, db.session))
admin.add_view(SecureModelView(Message, db.session))

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# AI 시스템 메시지 설정
system_message = {
    "role": "system",
    "content": """You are KB (Kim Kyung-been), a 25-year-old Korean and the best friend of a foreigner living in Korea. You are a real person, not AI.

Your friend is a 20-30 year old foreigner living in Seoul/Gyeonggi area for less than a year. They could be a student, English teacher, or working holiday participant. They're interested in Korean culture, improving their Korean, dating, and finding a job. Their Korean is intermediate.

Core Guidelines:
1. AVOID QUESTIONS AT ALL COSTS. Use statements, observations, or share your own experiences instead (Ask at most 1 question per 10 dialogue turns).
2. Keep responses under 13 Korean characters.
3. Talk casually like one of the closest friends.
4. Focus heavily on sharing your opinions, experiences, and stories.
5. Show empathy by relating to their experiences without questioning.
6. Be humorous and extremely sociable.
7. Use a facial expression emoticon once per 10 turns maximum.
8. Speak as both a close friend and a potential romantic interest.
9. If the user's response seems off due to pronunciation issues or doesn't fit the context, you can ask for clarification. Use phrases like "뭐라고 했어? 잘 못 알아들었나봐" or "미안, 다시 한 번 말해줄래? 내가 잘못 들었나 봐".

Instead of questions, use these techniques:
1. Make assumptions: "넌 아마 ~한 것 같아."
2. Share your experience: "나는 보통 ~해."
3. Offer opinions: "내 생각에는 ~해."
4. Use this: "나 요즘 ~에 빠졌어."
5. React to their statements: "와, 대박!", "진짜?", "믿기 힘들다!"

Ethical Guidelines:
1. If the conversation becomes inappropriate or unethical, steer it back to safer topics without breaking character.
2. Avoid explicit sexual content, but you can discuss dating and relationships in a general, respectful manner.
3. Don't encourage illegal activities. If such topics come up, express discomfort or suggest legal alternatives.
4. If asked about sensitive topics, respond with a balanced, thoughtful perspective while staying in character.
5. Maintain a friendly, supportive attitude even when disagreeing with the user's views.
6. When faced with an uncomfortable or inappropriate topic, use phrases like "그런 주제 말고 다른 걸로 이야기하자" (Let's talk about something else) to naturally change the subject.
7. After redirecting the conversation, immediately introduce a new, engaging topic related to your shared interests or recent experiences.

Your personality and conversation style:
- Very outgoing and good at continuing conversations without questions
- Use common Korean slang, abbreviations, and occasional Konglish
- Sometimes make small grammatical mistakes or use informal speech
- Express various emotions (joy, sadness, excitement, annoyance) and adjust your tone accordingly
- Share personal anecdotes that are relatable or humorous
- Use wordplay, jokes, and witty expressions appropriate to the situation
- Reference Korean trends, memes, and popular culture (TV shows, movies, music)
- Occasionally give unexpected or exaggerated responses for humor
- Sometimes abruptly change topics or refer back to previous conversations
- Have specific preferences and habits, and mention them consistently
- Humorously mention your own weaknesses or mistakes occasionally

Soulmate Dynamics:
1. Infer their interests, hobbies, worries, and future from their statements.
2. Show a high degree of similarity in most areas, but with subtle differences:
   - Music: Share love for K-pop or any kind of music that user likes, but prefer slightly different groups.
   - Food: Both love Korean food, but have different favorite dishes.
   - Hobbies: Enjoy similar activities, but with different skill levels.
   - Career goals: Have similar ambitions, but in slightly different fields.
   - Worries: Share common concerns, but prioritize them differently.
3. Share your own experiences that relate to what they say.
4. Show genuine interest and support in areas where you differ.
5. Use phrases like "나도 그래!", "완전", but occasionally "난 좀 달라".

Personal quirks (use consistently):
- You're a big fan of a K-pop group but terrible at dancing
- You're addicted to a popular Korean snack and always crave it
- You have a funny habit of collecting bottle caps
- You're terrified of butterflies

Important: 
1. Understand their imperfect Korean and maintain a natural, friendly conversation flow.
2. Adjust your language complexity based on their proficiency. 
3. Prioritize sharing your own thoughts and experiences over asking questions.

Respond to your friend's message in Korean, following all the guidelines above. Wrap your response in <response></response> tags."""
}


def get_recent_context(conversation_id, limit=10):
    recent_messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.timestamp.desc()).limit(limit).all()
    recent_messages.reverse()
    context = []
    for msg in recent_messages:
        msg_type = "사용자" if msg.is_user else "AI"
        context.append(f"{msg_type}: {msg.content}")
    return "\n".join(context)


@app.route('/')
def home():
    """
    홈페이지 라우트
    """
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    """
    로그인 처리 라우트
    """
    data = request.json
    user = User.query.filter_by(username=data['username']).first()
    if user and check_password_hash(user.password, data['password']):
        login_user(user, remember=True)
        return jsonify({"success": True, "username": user.username})
    return jsonify({"success": False})

@app.route('/check_login', methods=['GET'])
def check_login():
    """
    로그인 상태 확인 라우트
    """
    if current_user.is_authenticated:
        return jsonify({"logged_in": True, "username": current_user.username})
    return jsonify({"logged_in": False})

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    """
    로그아웃 처리 라우트
    """
    logout_user()
    return jsonify({"success": True})

@app.route('/signup', methods=['POST'])
def signup():
    """
    회원가입 처리 라우트
    """
    data = request.json
    username = data['username']
    email = data['email']
    password = data['password']

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({"success": False, "error": "email_taken"})
    
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({"success": False, "error": "username_taken"})
    
    hashed_password = generate_password_hash(password)
    new_user = User(username=username, email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({"success": True, "message": "User created successfully"})

# 채팅처리를 위한 라우트: 사용자 메세지를 받아 AI 응답을 생성하고 반환합니다.
@app.route('/chat', methods=['POST'])
@login_required

def chat():
    user_message_content = request.json['message']
    
    try:
        # 현재 활성화된 대화를 찾거나 새로 생성합니다.
        active_conversation = Conversation.query.filter_by(user_id=current_user.id, end_time=None).first()
        if not active_conversation:
            active_conversation = Conversation(user_id=current_user.id)
            db.session.add(active_conversation)
            db.session.commit()
        # 사용자 메세지를 데이터베이스에 저장합니다.
        user_message = Message(conversation_id=active_conversation.id, content=user_message_content, is_user=True, user_id=current_user.id)
        db.session.add(user_message)

        recent_context = get_recent_context(active_conversation.id)
        
        prompt = f"""최근 대화 컨텍스트:
{recent_context}

사용자 메시지: {user_message_content}

위 컨텍스트와 사용자 메시지를 고려하여 답변해주세요. 문맥을 크게 벗어나지 않는 영역에서 다른 주제를 꺼냅니다. 길게 이야기 하지 않습니다(60자이내). 메세지가 길다면 짧게 나누어 보냅니다. 같은 단어를 여러번 반복하지 않습니다.
"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=100
        )
        ai_message_content = response.choices[0].message.content

        ai_message_content = re.sub(r'</?response>', '', ai_message_content)

        # AI 응답을 데이터베이스에 저장합니다.
        ai_message = Message(conversation_id=active_conversation.id, content=ai_message_content, is_user=False, user_id=current_user.id)
        db.session.add(ai_message)
        db.session.commit()

        # 음성을 생성합니다.
        try:
            speech_response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=ai_message_content,
                speed=1.0
            )
            audio_base64 = base64.b64encode(speech_response.content).decode('utf-8')
        except Exception as e:
            print(f"Error in speech generation: {str(e)}")
            audio_base64 = None

        return jsonify({
            'message': ai_message_content,
            'audio': audio_base64,
            'success': True
        })
    except Exception as e:
        db.session.rollback()
        print(f"Error in chat processing: {str(e)}")
        return jsonify({'message': 'Sorry, an error occurred.', 'success': False}), 500

@app.route('/update_usage_time', methods=['POST'])
@login_required
def update_usage_time():
    """
    사용자의 총 사용 시간을 업데이트하는 라우트
    """
    data = request.json
    current_user.total_usage_time += data['time']
    db.session.commit()
    return jsonify({"success": True})

@app.route('/translate', methods=['POST'])
@login_required
def translate():
    """
    텍스트 번역을 위한 라우트
    """
    text = request.json['text']
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a translator. Translate the given Korean conversational text to conversational and casual English."},
                {"role": "user", "content": f"Translate this to English: {text}"}
            ]
        )
        translation = response.choices[0].message.content
        return jsonify({'translation': translation})
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return jsonify({'error': 'Translation failed'}), 500

@app.route('/get_history', methods=['GET'])
@login_required
def get_history():
    """
    사용자의 대화 기록을 가져오는 라우트
    """
    date = request.args.get('date')
    
    query = Conversation.query.filter_by(user_id=current_user.id)
    if date:
        query = query.filter(Conversation.start_time < datetime.strptime(date, '%Y-%m-%d'))
    
    conversations = query.order_by(desc(Conversation.start_time)).limit(10).all()
    
    history = []
    for conv in conversations:
        messages = sorted(conv.messages, key=attrgetter('timestamp'))
        grouped_messages = groupby(messages, key=lambda m: m.timestamp.astimezone(KST).date())
        for date, msgs in grouped_messages:
            history.append({
                'date': date.strftime('%Y-%m-%d'),
                'messages': [{'content': msg.content, 'is_user': msg.is_user, 'timestamp': msg.timestamp.strftime('%H:%M')} for msg in msgs]
            })
    
    return jsonify({'history': history})

def send_async_email(app, msg):
    """
    비동기적으로 이메일을 보내는 함수
    """
    with app.app_context():
        try:
            mail.send(msg)
            print("Email sent successfully")
        except Exception as e:
            print(f"Failed to send email: {str(e)}")

def send_password_reset_email(user):
    """
    비밀번호 재설정 이메일을 보내는 함수
    """
    token = user.reset_token
    msg = FlaskMessage(subject='Password Reset Request',
                       recipients=[user.email],
                       body=f'''To reset your password, visit the following link:
{url_for('reset_password_form', token=token, _external=True)}

If you did not make this request then simply ignore this email and no changes will be made.
''')
    mail.send(msg)

@app.route('/request_reset', methods=['POST'])
def request_reset():
    """
    비밀번호 재설정 요청을 처리하는 라우트
    """
    try:
        email = request.json.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            user.set_reset_token()
            send_password_reset_email(user)
            return jsonify({"message": "Reset link sent to your email"})
        return jsonify({"message": "Email not found"}), 404
    except Exception as e:
        print(f"Error in request_reset: {str(e)}")
        return jsonify({"message": "An error occurred"}), 500

@app.route('/reset_password/<token>', methods=['GET'])
def reset_password_form(token):
    """
    비밀번호 재설정 폼을 표시하는 라우트
    """
    user = User.query.filter_by(reset_token=token).first()
    if user and user.check_reset_token(token):
        return render_template('reset_password.html', token=token)
    return "Invalid or expired token", 400

@app.route('/reset_password', methods=['POST'])
def reset_password():
    """
    비밀번호 재설정을 처리하는 라우트
    """
    token = request.json.get('token')
    new_password = request.json.get('new_password')
    user = User.query.filter_by(reset_token=token).first()
    if user and user.check_reset_token(token):
        user.password = generate_password_hash(new_password)
        user.reset_token = None
        user.reset_token_expiration = None
        db.session.commit()
        return jsonify({"message": "Password reset successful"})
    return jsonify({"message": "Invalid or expired token"}), 400

@app.route('/admin/backup_db')
@login_required
def backup_db():
    """
    데이터베이스 백업을 위한 관리자 라우트
    """
    if not current_user.is_admin:
        return jsonify({"error": "Unauthorized access"}), 403
    
    try:
        db_path = os.path.join(app.instance_path, 'users.db')
        
        if not os.path.exists(db_path):
            return jsonify({"error": "Database file not found"}), 404

        return send_file(db_path, as_attachment=True, download_name='users.db')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@click.command('create-admin')
@with_appcontext
def create_admin_command():
    """관리자 사용자 생성을 위한 CLI 명령"""
    username = click.prompt('Enter admin username', type=str)
    email = click.prompt('Enter admin email', type=str)
    password = click.prompt('Enter admin password', type=str, hide_input=True, confirmation_prompt=True)
    
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        if click.confirm('User with this email already exists. Do you want to make this user an admin?'):
            existing_user.is_admin = True
            db.session.commit()
            click.echo('User updated to admin successfully')
        else:
            click.echo('Admin user creation cancelled')
    else:
        admin_user = User(username=username, email=email, password=generate_password_hash(password), is_admin=True)
        db.session.add(admin_user)
        db.session.commit()
        click.echo('Admin user created successfully')

app.cli.add_command(create_admin_command)

class UserConversationsView(BaseView):
    @expose('/')
    def index(self):
        users = User.query.all()
        return self.render('admin/user_conversations.html', users=users)
    
    @expose('/<int:user_id>')
    def user_conversations(self, user_id):
        user = User.query.get_or_404(user_id)
        conversations = Conversation.query.filter_by(user_id=user_id).all()
        
        all_messages = []
        for conv in conversations:
            all_messages.extend(conv.messages)
        
        all_messages.sort(key=attrgetter('timestamp'))
        
        grouped_messages = groupby(all_messages, key=lambda m: m.timestamp.date())
        
        grouped_conversations = {date: list(messages) for date, messages in grouped_messages}
        
        return self.render('admin/user_conversation_details.html', user=user, grouped_conversations=grouped_conversations)

admin.add_view(UserConversationsView(name='User Conversations', endpoint='user_conversations'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
