async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Formulate the system of equations for Aya's walking and coffee-shop problem: 9/s + t/60 = 4 and 9/(s+2) + t/60 = 2.4, with context from taskInfo"
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, formulating system of equations, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    candidate_solutions = []
    for i in range(3):
        answer_generate_instruction = f"Subtask 2.{i+1}: Generate initial candidate numerical solutions for s and t by applying iterative reasoning approach, iteration {i+1}"
        answer_generate_desc = {
            'instruction': answer_generate_instruction,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results2 = await self.answer_generate(
            subtask_id=f"subtask_2_{i+1}",
            cot_agent_desc=answer_generate_desc
        )
        agents.append(f"AnswerGenerate agent {results2['cot_agent'].id}, iteration {i+1}, generating candidate solutions, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
        sub_tasks.append(f"Subtask 2.{i+1} output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])
        candidate_solutions.append(results2['answer'].content)

    cot_sc_instruction3 = "Subtask 3: Solve the two equations to find s and t, then compute the total time (in minutes) when walking at speed s + 0.5, considering candidate solutions"
    cot_sc_desc = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer']] + candidate_solutions,
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results3['list_thinking']):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, solving equations and computing total time, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    aggregate_instruction4 = "Subtask 4: Aggregate candidate total times from Subtask 3 and select the most consistent and accurate total time"
    aggregate_desc = {
        'instruction': aggregate_instruction4,
        'input': [taskInfo] + results3['list_answer'],
        'temperature': 0.0,
        'context': ["user query", "candidate total times from subtask 3"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, aggregating candidate total times, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    review_instruction5 = "Subtask 5: Review the aggregated total time for accuracy, consistency, and completeness"
    review_desc = {
        'instruction': review_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5 = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results5['review_agent'].id}, reviewing aggregated total time, feedback: {results5['thinking'].content}; correct: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: feedback - {results5['thinking'].content}; correct - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    formatter_instruction6 = "Subtask 6: Format the computed total time as an integer number of minutes and validate its accuracy"
    formatter_desc = {
        'instruction': formatter_instruction6,
        'input': [taskInfo, results4['thinking'], results4['answer'], results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4", "feedback of subtask 5", "correct of subtask 5"],
        'format': 'short and concise, without explanation'
    }
    results6 = await self.specific_format(
        subtask_id="subtask_6",
        formatter_desc=formatter_desc
    )
    agents.append(f"SpecificFormat agent {results6['formatter_agent'].id}, formatting final total time, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'], sub_tasks, agents)
    return final_answer, logs
