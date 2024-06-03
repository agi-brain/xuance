__import__('warnings').filterwarnings('ignore')


import functools as f
import re
import typing as t

import bs4
import pandas as pd
import requests


Cookies = t.Dict[str, str]
Grades = t.List[t.Dict[str, t.Any]]


class GpaInfo(t.TypedDict):
    绩点: str
    排名: str
    学期: t.List[t.Tuple[str, str]]


class TIS:
    _request_kwargs = {
        'headers': {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'},
        'verify': False,
    }

    def __init__(self, cookies: Cookies) -> None:
        self._cookies = cookies

    @classmethod
    def login(cls, username: str, password: str) -> 'TIS':
        url = 'https://cas.sustech.edu.cn/cas/login'
        params = {'service': 'https://tis.sustech.edu.cn/cas'}
        with requests.session() as session:
            response = session.get(url, params=params, **cls._request_kwargs)
            soup = bs4.BeautifulSoup(response.content, 'html.parser')
            execution = soup.select('input[name$="execution"]')[0]['value']
            data = {
                'username': username, 'password': password,
                'execution': execution, '_eventId': 'submit',
            }
            session.post(url, params=params, data=data, **cls._request_kwargs)
        return cls(dict(session.cookies))

    @f.cached_property
    def grades(self) -> pd.DataFrame:
        url = 'https://tis.sustech.edu.cn/cjgl/grcjcx/grcjcx'
        data = {
            'current': 1, 'cxbj': -1, 'kcmc': None,
            'pageSize': 100, 'pylx': 2, 'xn': None, 'xq': None,
        }
        response = requests.post(url, json=data, cookies=self._cookies, **self._request_kwargs)
        data = response.json()['content']['list']
        keys = {
            'kcdm': '代码', 'kcmc': '名称', 'kclb': '类别', 'yxmc': '院系',
            'xszscj': '成绩', 'xscj': '等级', 'xf': '学分',
        }
        df = pd.DataFrame([
            tuple(grade[key] for key in keys) for grade in data
        ])
        df.columns = pd.MultiIndex.from_tuples(keys.items())
        return df

    @f.cached_property
    def gpa_info(self) -> GpaInfo:
        url = 'https://tis.sustech.edu.cn/cjgl/xscjgl/xsgrcjcx/queryXnAndXqXfj'
        data = requests.post(url, cookies=self._cookies, **self._request_kwargs).json()
        strize = lambda x: f'{x:.2f}'
        return {
            '绩点': strize(data['xfjandpm']['PJXFJ']),
            '排名': data['xfjandpm']['PM'],
            '学期': [
                (item['XNXQ'], strize(item['XQXFJ']))
                for item in data['xnanxqxfj']
            ],
        }

    @property
    def gpa(self) -> float:
        grade_credit = [
            (float(row['xszscj'][0]), float(row['xf'][0]))
            for _, row in self.grades.iterrows()
            if row['xszscj'][0] is not None and row['xscj'][0] != 'P'
        ]
        credits = sum(c for _, c in grade_credit)
        if credits == 0:
            return 0.0
        return sum([self._map(g)*c for g, c in grade_credit]) / credits

    @property
    def gpa_str(self) -> str:
        return self.gpa_info['绩点']

    @property
    def sa(self) -> float:
        '''Score Average'''
        grade_credit = [
            (float(row['xszscj'][0]), float(row['xf'][0]))
            for _, row in self.grades.iterrows()
            if row['xszscj'][0] is not None
        ]
        credits = sum(c for _, c in grade_credit)
        if credits == 0:
            return 0.0
        return sum([g*c for g, c in grade_credit]) / credits

    @property
    def sa_str(self) -> str:
        return f'{self.sa:.2f}'

    @property
    def rank(self) -> str:
        '''Rank'''
        return self.gpa_info['排名']

    def _map(self, grade: int) -> float:
        for (boundary, value) in [
            # 研究生
            (95, 4.0), (90, 3.7), (85, 3.3), (80, 3.0), (77, 2.7),
            (73, 2.3), (70, 2.0), (67, 1.7), (63, 1.3), (60, 1.0),
            # 本科生
            # (97, 4.00), (93, 3.94), (90, 3.85), (87, 3.73), (83, 3.55),
            # (80, 3.32), (77, 3.09), (73, 2.78), (70, 2.42), (67, 2.08),
            # (63, 1.63), (60, 1.15),
        ]:
            if grade >= boundary:
                return value
        return 0.0


if __name__ == '__main__':
    # username='12332649'
    # password='0Cc010212'
    tis = TIS.login(input('(username) >>> '), input('(password) >>> '))
    # tis = TIS.login(input(username), input(password))
    print(f'Rank: {tis.rank}')
    print(f'GPA:   {tis.gpa_str} ~= {tis.gpa}')
    print(f'SA:   {tis.sa_str} ~= {tis.sa}')
    print(
        tis.grades
            .droplevel(0, axis=1)
            .sort_values(by=['成绩', '学分'], ascending=False)
            .to_markdown(index=False)
    )
