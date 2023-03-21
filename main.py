import telebot
from telebot import types
from telebot.types import ReplyKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from pprint import pprint
from model import load_model, clean_data
import pandas as pd
import os

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')



def run_bot(model):

    bot = telebot.TeleBot(TELEGRAM_TOKEN)

    data = {}

    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        msg = bot.reply_to(
            message, """Welcome to Lets Hire Bot! We predict the probability of a person being hired for a job position using advanced data analysis. Our bot helps HR professionals make better hiring decisions quickly and accurately. Input /predict to ask you for some data.""")

    @bot.message_handler(commands=['predict'])
    def start_prediction(message):
        msg = bot.reply_to(
            message, 'Welcome, state your city code in the format "city_{$number}"')
        bot.register_next_step_handler(msg, process_city_code)

    def process_city_code(message):
        city = message.text
        data['city'] = city
        msg = bot.reply_to(
            message, 'Insert  city development index (real number in range[0,1]) ')
        bot.register_next_step_handler(msg, process_development_index)

    def process_development_index(message):
        dev_index = message.text
        dev_index = dev_index.replace(",", ".")
        data['city_development_index'] = dev_index
        markup = types.ReplyKeyboardMarkup(
            one_time_keyboard=True, resize_keyboard=True)
        markup.add('Male', 'Female', 'Other')
        msg = bot.reply_to(message, "Gender ", reply_markup=markup)
        bot.register_next_step_handler(msg, process_gender)

    def process_gender(message):
        gender = message.text
        data['gender'] = gender
        markup = types.ReplyKeyboardMarkup(
            one_time_keyboard=True, resize_keyboard=True)
        markup.add('Has relevent experience', 'No relevent experience')
        msg = bot.reply_to(message, "Relevant experience ",
                           reply_markup=markup)
        bot.register_next_step_handler(msg, process_relevant_exp)

    def process_relevant_exp(message):
        rexp = message.text
        data['relevent_experience'] = rexp
        markup = types.ReplyKeyboardMarkup(
            one_time_keyboard=True, resize_keyboard=True)
        markup.add("no_enrollment", "Full time course", "Part time course")
        msg = bot.reply_to(message, "University enrollment ",
                           reply_markup=markup)
        bot.register_next_step_handler(msg, process_enrolment)

    def process_enrolment(message):
        experience = message.text
        data['enrolled_university'] = experience
        markup = types.ReplyKeyboardMarkup(
            one_time_keyboard=True, resize_keyboard=True)
        markup.add('Graduate', 'Masters', 'High School',
                   'Phd', 'Primary School')
        msg = bot.reply_to(message, "Education Level ", reply_markup=markup)
        bot.register_next_step_handler(msg, process_education_level)

    def process_education_level(message):
        enrollment = message.text
        data['education_level'] = enrollment
        markup = types.ReplyKeyboardMarkup(
            one_time_keyboard=True, resize_keyboard=True)
        markup.add('STEM', 'Business Degree', 'Arts',
                   'Humanities', 'No Major', 'Other')
        msg = bot.reply_to(message, "Major discipline", reply_markup=markup)
        bot.register_next_step_handler(msg, process_education_major)

    def process_education_major(message):
        education = message.text
        data['major_discipline'] = education
        markup = types.ReplyKeyboardMarkup(
            one_time_keyboard=True, resize_keyboard=True)
        markup.add('<10', '10/49', '50-99', '100-500',
                   '500-999', '1000-4999', '5000-9999', '10000+')
        msg = bot.reply_to(message, "Company size ", reply_markup=markup)
        bot.register_next_step_handler(msg, process_company_size)

    def process_company_size(message):
        size = message.text
        data['company_size'] = size
        msg = bot.reply_to(message, 'Years of experience"')
        bot.register_next_step_handler(msg, process_time_experience)

    def process_time_experience(message):
        experience = message.text
        data['experience'] = experience
        markup = types.ReplyKeyboardMarkup(
            one_time_keyboard=True, resize_keyboard=True)
        markup.add('Pvt Ltd', 'Funded Startup',
                   'Early Stage Startup', 'Other', 'Public Sector', 'NGO')
        msg = bot.reply_to(message, "Company type ", reply_markup=markup)
        bot.register_next_step_handler(msg, process_company_type)

    def process_company_type(message):
        c_type = message.text
        data['company_type'] = c_type
        msg = bot.reply_to(message, 'Years in last job:')
        bot.register_next_step_handler(msg, process_last_new_job)

    def process_last_new_job(message):
        last_new_job = message.text
        data['last_new_job'] = last_new_job
        msg = bot.reply_to(message, 'Training hours completed:')
        bot.register_next_step_handler(msg, process_training_hours)

    def process_training_hours(message):
        hours = message.text
        data['training_hours'] = hours
        msg = bot.reply_to(message, predict_with_model())

    def predict_with_model():
        df = pd.DataFrame([data])
        X = clean_data(df)
        Y = model.predict(X)
        prob = model.predict_proba(X)

        return f'Your probability of staying with us is {prob[0, 1]}' + ". If you want to start over input /predict again."

    bot.polling(none_stop=True, interval=0)


if __name__ == '__main__':
    model = load_model()

    run_bot(model)
