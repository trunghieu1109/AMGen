async def forward_24(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction2 = "Subtask 1: Parse the three given log_2 equations and label them as equations (1), (2), (3)"
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, parsing log equations, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_instruction1 = "Subtask 2: Rewrite each log equation in its equivalent exponential form to eliminate the logarithms"
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results1 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, rewriting log equations to exponential form, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction3 = "Subtask 3: Translate the exponential forms into a linear system in variables Lx=log_2 x, Ly=log_2 y, Lz=log_2 z"
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results3['list_thinking']):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, translating exponential forms to linear system, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    candidate_solutions = []
    for i in range(1):
        cot_sc_instruction4 = "Subtask 4: Generate candidate numeric solutions for (Lx, Ly, Lz) by solving pairs of linear equations"
        cot_sc_desc4 = {
            'instruction': cot_sc_instruction4,
            'input': [taskInfo, results3['thinking'], results3['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
        }
        results4 = await self.sc_cot(
            subtask_id="subtask_4",
            cot_sc_desc=cot_sc_desc4,
            n_repeat=self.max_sc
        )
        for idx, key in enumerate(results4['list_thinking']):
            agents.append(f"CoT-SC agent {results4['cot_agent'][idx].id}, generating candidate numeric solutions, thinking: {results4['list_thinking'][idx]}; answer: {results4['list_answer'][idx]}")
        sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
        logs.append(results4['subtask_desc'])
        candidate_solutions.append(results4['answer'].content)

    aggregate_instruction5 = "Subtask 5: Consolidate all candidate (Lx, Ly, Lz) sets and identify the unique consistent solution"
    aggregate_desc5 = {
        'instruction': aggregate_instruction5,
        'input': [taskInfo] + candidate_solutions,
        'temperature': 0.0,
        'context': ["user query", "candidate solutions from subtask 4"]
    }
    results5 = await self.aggregate(
        subtask_id="subtask_5",
        aggregate_desc=aggregate_desc5
    )
    agents.append(f"Aggregate agent {results5['aggregate_agent'].id}, consolidating candidate solutions, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    review_instruction6 = "Subtask 6: Validate the unique (Lx, Ly, Lz) solution against all three original equations for correctness"
    review_desc6 = {
        'instruction': review_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"]
    }
    results6 = await self.review(
        subtask_id="subtask_6",
        review_desc=review_desc6
    )
    agents.append(f"Review agent {results6['review_agent'].id}, validating unique solution, feedback: {results6['thinking'].content}; correct: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: feedback - {results6['thinking'].content}; correct - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    if "valid" in results6['answer'].content.lower():
        debate_instruction7 = "Subtask 7: Convert the validated (Lx, Ly, Lz) solution into the value of |log_2(x^4 y^3 z^2)| and calculate m+n where m/n is the simplified fraction"
        final_decision_instruction7 = "Subtask 7: Make final decision on the value of m+n"
        debate_desc7 = {
            'instruction': debate_instruction7,
            'context': ["user query", results6['thinking'].content, results6['answer'].content],
            'input': [taskInfo, results6['thinking'], results6['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.5
        }
        final_decision_desc7 = {
            'instruction': final_decision_instruction7,
            'output': ["thinking", "answer"],
            'temperature': 0.0
        }
        results7 = await self.debate(
            subtask_id="subtask_7",
            debate_desc=debate_desc7,
            final_decision_desc=final_decision_desc7,
            n_repeat=self.max_round
        )
        for round in range(self.max_round):
            for idx, agent in enumerate(results7['debate_agent']):
                agents.append(f"Debate agent {agent.id}, round {round}, converting solution and calculating final output, thinking: {results7['list_thinking'][round][idx].content}; answer: {results7['list_answer'][round][idx].content}")
        agents.append(f"Final Decision agent, calculating final output, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
        sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
        logs.append(results7['subtask_desc'])

        formatter_instruction8 = "Subtask 8: Format the final integer result m+n according to the problem's output requirements"
        formatter_desc8 = {
            'instruction': formatter_instruction8,
            'input': [taskInfo, results7['thinking'], results7['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 7", "answer of subtask 7"],
            'format': 'short and concise, without explanation'
        }
        results8 = await self.specific_format(
            subtask_id="subtask_8",
            formatter_desc=formatter_desc8
        )
        agents.append(f"SpecificFormat agent {results8['formatter_agent'].id}, formatting final integer result, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
        sub_tasks.append(f"Subtask 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
        logs.append(results8['subtask_desc'])

        final_answer = await self.make_final_answer(results8['thinking'], results8['answer'], sub_tasks, agents)
        return final_answer, logs

    else:
        cot_instruction9 = "Subtask 9: Clarify and validate all domain assumptions (x,y,z>0) and detect inconsistencies"
        cot_agent_desc9 = {
            'instruction': cot_instruction9,
            'input': [taskInfo, results6['thinking'], results6['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
        }
        results9 = await self.cot(
            subtask_id="subtask_9",
            cot_agent_desc=cot_agent_desc9
        )
        agents.append(f"CoT agent {results9['cot_agent'].id}, clarifying domain assumptions, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
        sub_tasks.append(f"Subtask 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
        logs.append(results9['subtask_desc'])

        cot_instruction10 = "Subtask 10: Identify any missing reasoning steps if positivity fails and generate necessary corrections"
        cot_agent_desc10 = {
            'instruction': cot_instruction10,
            'input': [taskInfo, results9['thinking'], results9['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 9", "answer of subtask 9"]
        }
        results10 = await self.answer_generate(
            subtask_id="subtask_10",
            cot_agent_desc=cot_agent_desc10
        )
        agents.append(f"AnswerGenerate agent {results10['cot_agent'].id}, identifying missing reasoning steps and corrections, thinking: {results10['thinking'].content}; answer: {results10['answer'].content}")
        sub_tasks.append(f"Subtask 10 output: thinking - {results10['thinking'].content}; answer - {results10['answer'].content}")
        logs.append(results10['subtask_desc'])

        final_answer = await self.make_final_answer(results10['thinking'], results10['answer'], sub_tasks, agents)
        return final_answer, logs
