import time
from instapy import InstaPy
import clarifai
print(clarifai.__file__)


# --- InstaPy Class ---
class InstaPyBot:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.session = None

    def login(self):
        print(f"Logging in as {self.username}...")
        self.session = InstaPy(username=self.username, password=self.password)
        self.session.login()

    def get_followers(self, amount=10):
        print(f"Fetching {amount} followers for {self.username}...")
        followers = self.session.grab_followers(amount=amount, media=None, wait_for_download=True)
        print(f"Followers retrieved: {len(followers)}")
        return followers

    def get_posts(self, amount=5):
        print(f"Fetching {amount} posts for {self.username}...")
        posts = self.session.grab_posts(amount=amount)
        print(f"Posts retrieved: {len(posts)}")
        return posts

    def get_following(self):
        print(f"Fetching following list for {self.username}...")
        following = self.session.grab_following()
        print(f"Following list: {len(following)}")
        return following

    def like_post(self, media_id):
        print(f"Liking post with ID {media_id}...")
        self.session.like_by_tags(media_id, amount=1)

    def comment_post(self, media_id, text):
        print(f"Commenting on post with ID {media_id}...")
        self.session.comment_by_tags(media_id, text)


# --- Running the Bot ---
username = "liam_x.r"  # Replace with real username
password = "H_R2D44Y_ZbETdM"  # Replace with real password

bot = InstaPyBot(username, password)
bot.login()

# Simulate fetching followers
followers = bot.get_followers(amount=20)  # Change amount as needed

# Simulate fetching posts
posts = bot.get_posts(amount=5)  # Change amount as needed

# Simulate fetching following list
following = bot.get_following()

# Simulate liking a post and commenting
if posts:
    media_id = posts[0]  # Replace with actual media ID from posts
    bot.like_post(media_id)
    bot.comment_post(media_id, text="Great post!")
