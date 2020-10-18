import pyttsx3

text = pyttsx3.init()

# 팔꿈치 각도
pushup_angle = 100
# 무릎 각도
sqaut_angle = 60
# 상체 높이
upper_body_height = 90
# 하체 높이(엉덩이 기준)
under_body_height = 70

# 잘못된 푸쉬업 자세에 대한 음성
def pushup_voice():

    pushup_angle = 90
    text.say('잘못된 팔꿈치 각도입니다. 팔꿈치 각도를 %s도까지만 하세요' % pushup_angle)
    print('잘못된 각도입니다. 팔꿈치 각도를 %s도까지만 하세요' % pushup_angle)
    text.runAndWait()

    text.stop()

# 잘못된 스쿼트 자세에 대한 음성
def squat_voice():

    sqaut_angle = 90
    text.say('잘못된 무릎 각도입니다. 무릎 각도를 %s도까지만 내리세요' % sqaut_angle)
    print('잘못된 각도입니다. 무릎 각도를 %s도까지만 내리세요' % sqaut_angle)
    text.runAndWait()

    text.stop()

# 잘못된 플랭크 자세에 대한 음성
def plank_voice():

    text.say('자세가 잘못되었습니다. 상체와 엉덩이를 수평으로 유지하세요')
    print('자세가 잘못되었습니다. 상체와 엉덩이를 수평으로 유지하세요')
    text.runAndWait()

    text.stop()

if upper_body_height != under_body_height:
    plank_voice()
else:
    text.say('자세가 훌륭합니다. 복근 자극을 위해 복근에 힘을 주세요')
    print('자세가 훌륭합니다. 복근 자극을 위해 복근에 힘을 주세요')
    text.runAndWait()
if pushup_angle != 90:
    pushup_voice()
else:
    text.say('자세가 훌륭합니다. 가슴 자극을 위해 노력하세요')
    print('자세가 훌륭합니다. 가슴 자극을 위해 노력하세요')
    text.runAndWait()
if sqaut_angle > 90:
    text.say('둔근 강화를 위해 90도까지 내려가세요')
    print('둔근 강화를 위해 90도까지 내려가세요')
    text.runAndWait()
elif sqaut_angle < 90:
    squat_voice()
else:
    text.say('자세가 정확합니다. 무릎이 벌어지지 않게 주의하세요')
    print('자세가 정확합니다. 무릎이 벌어지지 않게 주의하세요')
    text.runAndWait()
