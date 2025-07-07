async def forward_24(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Parse the three given logarithmic equations and label them as equations (1), (2), and (3) with detailed explanation"
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, parsing logarithmic equations, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_reflect_instruction2 = "Subtask 2: Convert each logarithmic equation into its equivalent exponential form and verify the correctness of the conversion"
    critic_instruction2 = "Please review the exponential form conversion for correctness and identify any errors or inconsistencies"
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    critic_desc2 = {
        'instruction': critic_instruction2,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results2 = await self.reflexion(
        subtask_id="subtask_2",
        cot_reflect_desc=cot_reflect_desc2,
        critic_desc=critic_desc2,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, converting log equations to exponential form, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results2['list_feedback']))):
        agents.append(f"Critic agent {results2['critic_agent'].id}, feedback: {results2['list_feedback'][i].content}; correctness: {results2['list_correct'][i].content}")
        if i + 1 < len(results2['list_thinking']) and i + 1 < len(results2['list_answer']):
            agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining conversion, thinking: {results2['list_thinking'][i + 1].content}; answer: {results2['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_reflect_instruction3 = "Subtask 3: Derive the linear system in variables Lx=log_2 x, Ly=log_2 y, Lz=log_2 z from the exponential forms and verify the system's correctness"
    critic_instruction3 = "Please review the derived linear system for consistency with the original problem and identify any errors"
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
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
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, deriving linear system, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, feedback: {results3['list_feedback'][i].content}; correctness: {results3['list_correct'][i].content}")
        if i + 1 < len(results3['list_thinking']) and i + 1 < len(results3['list_answer']):
            agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining linear system, thinking: {results3['list_thinking'][i + 1].content}; answer: {results3['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_sc_instruction4 = "Subtask 4: Solve the linear system for (Lx, Ly, Lz) using multiple methods and explore alternative solutions"
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
    for idx in range(len(results4['list_thinking'])):
        agents.append(f"CoT-SC agent {results4['cot_agent'][idx].id}, solving linear system, thinking: {results4['list_thinking'][idx]}; answer: {results4['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    debate_instruction5 = "Subtask 5: Debate among agents to evaluate and consolidate candidate solutions for (Lx, Ly, Lz) to identify the most consistent and valid solution"
    final_decision_instruction5 = "Subtask 5: Make final decision on the unique consistent solution for (Lx, Ly, Lz)"
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", results4['thinking'].content, results4['answer'].content],
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc5 = {
        'instruction': final_decision_instruction5,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        final_decision_desc=final_decision_desc5,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results5['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, evaluating candidate solutions, thinking: {results5['list_thinking'][round][idx].content}; answer: {results5['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding unique solution, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    cot_reflect_instruction6 = "Subtask 6: Validate the unique solution (Lx, Ly, Lz) by substituting back into the original logarithmic equations to verify correctness"
    critic_instruction6 = "Please review the validation step and identify any inconsistencies or errors"
    cot_reflect_desc6 = {
        'instruction': cot_reflect_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"]
    }
    critic_desc6 = {
        'instruction': critic_instruction6,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results6 = await self.reflexion(
        subtask_id="subtask_6",
        cot_reflect_desc=cot_reflect_desc6,
        critic_desc=critic_desc6,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, validating solution against original equations, thinking: {results6['list_thinking'][0].content}; answer: {results6['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results6['list_feedback']))):
        agents.append(f"Critic agent {results6['critic_agent'].id}, feedback: {results6['list_feedback'][i].content}; correctness: {results6['list_correct'][i].content}")
        if i + 1 < len(results6['list_thinking']) and i + 1 < len(results6['list_answer']):
            agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, refining validation, thinking: {results6['list_thinking'][i + 1].content}; answer: {results6['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    if "valid" in results6['answer'].content.lower():
        cot_instruction7 = "Subtask 7: Calculate the value of |log_2(x^4 y^3 z^2)| using the validated (Lx, Ly, Lz) values and express it as a simplified fraction m/n"
        cot_agent_desc7 = {
            'instruction': cot_instruction7,
            'input': [taskInfo, results6['thinking'], results6['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
        }
        results7 = await self.cot(
            subtask_id="subtask_7",
            cot_agent_desc=cot_agent_desc7
        )
        agents.append(f"CoT agent {results7['cot_agent'].id}, calculating |log_2(x^4 y^3 z^2)|, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
        sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
        logs.append(results7['subtask_desc'])

        cot_instruction8 = "Subtask 8: Format the final integer result m+n according to the problem's output requirements, ensuring m and n are relatively prime"
        cot_agent_desc8 = {
            'instruction': cot_instruction8,
            'input': [taskInfo, results7['thinking'], results7['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 7", "answer of subtask 7"],
            'format': 'short and concise, without explanation'
        }
        results8 = await self.specific_format(
            subtask_id="subtask_8",
            formatter_desc=cot_agent_desc8
        )
        agents.append(f"SpecificFormat agent {results8['formatter_agent'].id}, formatting final integer result, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
        sub_tasks.append(f"Subtask 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
        logs.append(results8['subtask_desc'])

        final_answer = await self.make_final_answer(results8['thinking'], results8['answer'], sub_tasks, agents)
        return final_answer, logs

    else:
        cot_instruction9 = "Subtask 9: Clarify and validate all domain assumptions (x, y, z > 0) and detect any inconsistencies in the problem setup or solution"
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

        cot_instruction10 = "Subtask 10: Identify any missing reasoning steps or corrections needed if positivity or other assumptions fail, and generate necessary corrections"
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
