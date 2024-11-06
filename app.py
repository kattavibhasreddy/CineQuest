from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'CineQuest24'
db = SQLAlchemy(app)

# Database model for User
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    favorite_hero = db.Column(db.String(100))
    favorite_director = db.Column(db.String(100))
    favorite_genre = db.Column(db.String(100))
    recommended_movies = db.Column(db.Text)  # Store comma-separated movie titles

# Load movie dataset and prepare for recommendations
data = pd.read_csv('movie_details_total.csv')
data['combined_features'] = (
    data['Actors'].fillna('') + " " +
    data['Director'].fillna('') + " " +
    data['Genre'].fillna('') + " " +
    data['Plot'].fillna('')
)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create the database tables
with app.app_context():
    db.create_all()

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for user registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists!')
            return redirect(url_for('register'))
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

# Route for user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            session['user_id'] = user.id
            return redirect(url_for('set_preferences'))
        else:
            flash('Invalid username or password! If you don’t have an account, please register.')
            return redirect(url_for('login'))
    return render_template('login.html')

# Route for setting preferences
@app.route('/set_preferences', methods=['GET', 'POST'])
def set_preferences():
    if request.method == 'POST':
        user_id = session.get('user_id')
        favorite_hero = request.form['favorite_hero']
        favorite_director = request.form['favorite_director']
        favorite_genre = request.form['favorite_genre']
        user = User.query.get(user_id)

        if not user.favorite_hero:  # Only set preferences if they haven't been set
            user.favorite_hero = favorite_hero
            user.favorite_director = favorite_director
            user.favorite_genre = favorite_genre
            db.session.commit()

        return redirect(url_for('recommendations'))
    return render_template('set_preferences.html')

# Route for movie recommendations
from flask import session
@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    user_id = session.get('user_id')
    user = User.query.get(user_id)

    if user:
        # Clear existing recommendations if the "Get More Recommendations" button is clicked (POST request)
        if request.method == 'POST':
            session.pop('recommended_movies', None)

        # Generate recommendations if not already stored in the session
        if 'recommended_movies' not in session:
            user_input = f"{user.favorite_hero} {user.favorite_director} {user.favorite_genre}"
            user_input_vector = tfidf.transform([user_input])
            user_sim_scores = cosine_similarity(user_input_vector, tfidf_matrix)
            sim_scores = list(enumerate(user_sim_scores[0]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            recommended_movies = []
            already_recommended = set(user.recommended_movies.split(',')) if user.recommended_movies else set()

            for score in sim_scores:
                movie_index = score[0]
                movie_title = data['Title'].iloc[movie_index]
                if movie_title not in already_recommended:
                    recommended_movies.append(movie_title)
                    already_recommended.add(movie_title)
                    if len(recommended_movies) == 5:
                        break

            session['recommended_movies'] = recommended_movies
            user.recommended_movies = ','.join(already_recommended)
            db.session.commit()

        return render_template('recommendations.html', movies=session['recommended_movies'])
    return redirect(url_for('login'))


# Route to show movie details
@app.route('/movie_details/<movie_title>')
def movie_details(movie_title):
    movie = data[data['Title'] == movie_title].iloc[0]
    movie_info = {
        'title': movie['Title'],
        'actors': movie['Actors'],
        'plot': movie['Plot'],
        'genre': movie['Genre'],
        'director': movie['Director']
    }
    return render_template('movie_details.html', movie=movie_info)

@app.route('/change_preferences', methods=['GET', 'POST'])
def change_preferences():
    user_id = session.get('user_id')
    user = User.query.get(user_id)

    if request.method == 'POST':
        user.favorite_hero = request.form['favorite_hero']
        user.favorite_director = request.form['favorite_director']
        user.favorite_genre = request.form['favorite_genre']
        db.session.commit()
        session.pop('recommended_movies', None)  # Clear session recommendations
        return redirect(url_for('recommendations'))

    return render_template('change_preferences.html', user=user)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('recommended_movies', None)  # Clear session recommendations on logout
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
