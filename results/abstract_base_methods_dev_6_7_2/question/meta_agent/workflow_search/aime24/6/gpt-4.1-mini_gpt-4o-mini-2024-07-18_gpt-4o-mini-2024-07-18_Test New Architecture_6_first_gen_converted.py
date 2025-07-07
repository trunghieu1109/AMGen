async def forward_6(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Sub-task 1: Define variables for box edges a, b, c and write the volume and surface area equations V = abc = 23 and SA = 2(ab + bc + ca) = 54." 
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing volume and surface area equations, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction2 = "Sub-task 2: Derive the formula for the sphere radius r in terms of a, b, c: r = (1/2)√(a^2 + b^2 + c^2)."
    N = self.max_sc
    cot_sc_desc = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc,
        n_repeat=N
    )
    for idx, key in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, deriving sphere radius formula, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_reflect_instruction3 = "Sub-task 3: Set up the system of equations to find positive real solutions (a, b, c) satisfying the volume and surface area constraints."
    critic_instruction3 = "Please review the system of equations setup and provide its limitations."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc3 = {
        'instruction': critic_instruction3,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results3 = await self.reflexion(
        subtask_id="subtask_3",
        cot_reflect_desc=cot_reflect_desc3,
        critic_desc=critic_desc3,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, setting up system of equations, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, providing feedback, thinking: {results3['list_feedback'][i].content}; answer: {results3['list_correct'][i].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    candidate_solutions = []
    for i in range(1):
        cot_instruction4 = "Sub-task 4: Generate candidate dimension triples (a, b, c) by solving the polynomial equations or using a numerical root-finding approach."
        cot_agent_desc4 = {
            'instruction': cot_instruction4,
            'input': [taskInfo, results3['thinking'], results3['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
        }
        results4 = await self.cot(
            subtask_id="subtask_4",
            cot_agent_desc=cot_agent_desc4
        )
        agents.append(f"CoT agent {results4['cot_agent'].id}, generating candidate triples, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
        sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
        logs.append(results4['subtask_desc'])
        candidate_solutions.append(results4['answer'].content)

    aggregate_instruction5 = "Sub-task 5: Consolidate all real positive solutions for (a, b, c) into a single set of candidate boxes."
    aggregate_desc5 = {
        'instruction': aggregate_instruction5,
        'input': [taskInfo] + candidate_solutions,
        'temperature': 0.0,
        'context': ["user query", "solutions generated from subtask 4"]
    }
    results5 = await self.aggregate(
        subtask_id="subtask_5",
        aggregate_desc=aggregate_desc5
    )
    agents.append(f"CoT agent {results5['aggregate_agent'].id}, consolidating candidate solutions, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    programmer_instruction6 = "Sub-task 6: Validate each candidate triple against the original volume and surface area exactly equal to 23 and 54."
    programmer_desc6 = {
        'instruction': programmer_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"],
        'entry_point': "validate_candidates"
    }
    results6 = await self.programmer(
        subtask_id="subtask_6",
        programmer_desc=programmer_desc6
    )
    agents.append(f"Programmer Agent {results6['programmer_agent'].id}, validating candidates, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}, executing results: {results6['exec_result']}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}; output - {results6['exec_result']}")
    logs.append(results6['subtask_desc'])

    condition = 'valid solutions found' in results6['exec_result']

    if condition:
        cot_instruction7 = "Sub-task 7: Identify and clarify the units of a, b, c, V, SA, and r to ensure dimensional consistency."
        formatter_desc7 = {
            'instruction': cot_instruction7,
            'input': [taskInfo, results6['thinking'], results6['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 6", "answer of subtask 6"],
            'format': 'short and concise, without explanation'
        }
        results7 = await self.specific_format(
            subtask_id="subtask_7",
            formatter_desc=formatter_desc7
        )
        agents.append(f"CoT agent {results7['formatter_agent'].id}, clarifying units, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
        sub_tasks.append(f"Sub-task 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
        logs.append(results7['subtask_desc'])
    else:
        review_instruction8 = "Sub-task 8: Validate that r^2 simplifies to a rational p/q in lowest terms and identify p and q."
        review_desc8 = {
            'instruction': review_instruction8,
            'input': [taskInfo, results6['thinking'], results6['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
        }
        results8 = await self.review(
            subtask_id="subtask_8",
            review_desc=review_desc8
        )
        agents.append(f"Review agent {results8['review_agent'].id}, validating r^2 simplification, feedback: {results8['thinking'].content}; correct: {results8['answer'].content}")
        sub_tasks.append(f"Sub-task 8 output: feedback - {results8['thinking'].content}; correct - {results8['answer'].content}")
        logs.append(results8['subtask_desc'])

        formatter_instruction9 = "Sub-task 9: Enhance the clarity of the derivation and ensure coherent presentation of p, q, and the final p+q result."
        formatter_desc9 = {
            'instruction': formatter_instruction9,
            'input': [taskInfo, results8['thinking'], results8['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 8", "answer of subtask 8"],
            'format': 'short and concise, without explanation'
        }
        results9 = await self.specific_format(
            subtask_id="subtask_9",
            formatter_desc=formatter_desc9
        )
        agents.append(f"CoT agent {results9['formatter_agent'].id}, enhancing clarity, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
        sub_tasks.append(f"Sub-task 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
        logs.append(results9['subtask_desc'])

        cot_agent_instruction10 = "Sub-task 10: Format the final numeric answer p+q according to the required output spec."
        cot_agent_desc10 = {
            'instruction': cot_agent_instruction10,
            'input': [taskInfo, results9['thinking'], results9['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 9", "answer of subtask 9"]
        }
        results10 = await self.answer_generate(
            subtask_id="subtask_10",
            cot_agent_desc=cot_agent_desc10
        )
        agents.append(f"CoT agent {results10['cot_agent'].id}, formatting final answer, thinking: {results10['thinking'].content}; answer: {results10['answer'].content}")
        sub_tasks.append(f"Sub-task 10 output: thinking - {results10['thinking'].content}; answer - {results10['answer'].content}")
        logs.append(results10['subtask_desc'])

    final_answer = await self.make_final_answer(results10['thinking'], results10['answer'], sub_tasks, agents) if not condition else await self.make_final_answer(results7['thinking'], results7['answer'], sub_tasks, agents)
    return final_answer, logs
