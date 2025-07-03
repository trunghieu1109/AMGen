async def forward_190(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction1 = "Subtask 1: Analyze the first reaction: treatment of 3-(hydroxymethyl)-5-(prop-1-en-2-yl)cyclohexan-1-one with sodium hydride followed by benzyl bromide, and determine the structure of product 1."
    results1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_instruction=cot_sc_instruction1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query"],
        n_repeat=self.max_sc
    )
    cot_reflect_instruction1 = "Subtask 1 Reflexion: Based on the SC-CoT outputs, review and refine the structure determination of product 1."
    critic_instruction1 = "Please review the validity and limitations of the proposed structures for product 1."
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1, 'input': [taskInfo] + results1['list_thinking'] + results1['list_answer'], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query"] + results1['list_thinking'] + results1['list_answer']
    }
    critic_desc1 = {
        'instruction': critic_instruction1, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    results_reflex1 = await self.reflexion(
        subtask_id="subtask_1_reflexion",
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )
    agents.append(f"SC_CoT agents {[agent.id for agent in results1['cot_agent']]}, analyzing first reaction, thinking samples and answers collected.")
    for i in range(self.max_round):
        agents.append(f"Reflexion CoT agent {results_reflex1['cot_agent'].id}, round {i}, refining product 1 structure, thinking: {results_reflex1['list_thinking'][i].content}; answer: {results_reflex1['list_answer'][i].content}")
        agents.append(f"Critic agent {results_reflex1['critic_agent'].id}, round {i}, feedback: {results_reflex1['list_feedback'][i].content}; correctness: {results_reflex1['list_correct'][i].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results_reflex1['thinking'].content}; answer - {results_reflex1['answer'].content}")
    logs.append(results_reflex1['subtask_desc'])

    cot_sc_instruction2 = "Subtask 2: Analyze the second reaction: treatment of product 1 with p-toluenesulfonyl hydrazide in the presence of catalytic HCl, and determine the structure of product 2."
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_instruction=cot_sc_instruction2,
        input_list=[taskInfo, results_reflex1['thinking'], results_reflex1['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", results_reflex1['thinking'], results_reflex1['answer']],
        n_repeat=self.max_sc
    )
    cot_reflect_instruction2 = "Subtask 2 Reflexion: Based on SC-CoT outputs, review and refine the structure determination of product 2."
    critic_instruction2 = "Please review the validity and limitations of the proposed structures for product 2."
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2, 'input': [taskInfo, results_reflex1['thinking'], results_reflex1['answer']] + results2['list_thinking'] + results2['list_answer'], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query", results_reflex1['thinking'], results_reflex1['answer']] + results2['list_thinking'] + results2['list_answer']
    }
    critic_desc2 = {
        'instruction': critic_instruction2, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    results_reflex2 = await self.reflexion(
        subtask_id="subtask_2_reflexion",
        cot_reflect_desc=cot_reflect_desc2,
        critic_desc=critic_desc2,
        n_repeat=self.max_round
    )
    agents.append(f"SC_CoT agents {[agent.id for agent in results2['cot_agent']]}, analyzing second reaction, thinking samples and answers collected.")
    for i in range(self.max_round):
        agents.append(f"Reflexion CoT agent {results_reflex2['cot_agent'].id}, round {i}, refining product 2 structure, thinking: {results_reflex2['list_thinking'][i].content}; answer: {results_reflex2['list_answer'][i].content}")
        agents.append(f"Critic agent {results_reflex2['critic_agent'].id}, round {i}, feedback: {results_reflex2['list_feedback'][i].content}; correctness: {results_reflex2['list_correct'][i].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results_reflex2['thinking'].content}; answer - {results_reflex2['answer'].content}")
    logs.append(results_reflex2['subtask_desc'])

    cot_sc_instruction3 = "Subtask 3: Analyze the third reaction: treatment of product 2 at low temperature with n-butyllithium followed by aqueous ammonium chloride, and determine the structure of product 3."
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_instruction=cot_sc_instruction3,
        input_list=[taskInfo, results_reflex2['thinking'], results_reflex2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", results_reflex2['thinking'], results_reflex2['answer']],
        n_repeat=self.max_sc
    )
    cot_reflect_instruction3 = "Subtask 3 Reflexion: Based on SC-CoT outputs, review and refine the structure determination of product 3."
    critic_instruction3 = "Please review the validity and limitations of the proposed structures for product 3."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3, 'input': [taskInfo, results_reflex2['thinking'], results_reflex2['answer']] + results3['list_thinking'] + results3['list_answer'], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query", results_reflex2['thinking'], results_reflex2['answer']] + results3['list_thinking'] + results3['list_answer']
    }
    critic_desc3 = {
        'instruction': critic_instruction3, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    results_reflex3 = await self.reflexion(
        subtask_id="subtask_3_reflexion",
        cot_reflect_desc=cot_reflect_desc3,
        critic_desc=critic_desc3,
        n_repeat=self.max_round
    )
    agents.append(f"SC_CoT agents {[agent.id for agent in results3['cot_agent']]}, analyzing third reaction, thinking samples and answers collected.")
    for i in range(self.max_round):
        agents.append(f"Reflexion CoT agent {results_reflex3['cot_agent'].id}, round {i}, refining product 3 structure, thinking: {results_reflex3['list_thinking'][i].content}; answer: {results_reflex3['list_answer'][i].content}")
        agents.append(f"Critic agent {results_reflex3['critic_agent'].id}, round {i}, feedback: {results_reflex3['list_feedback'][i].content}; correctness: {results_reflex3['list_correct'][i].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results_reflex3['thinking'].content}; answer - {results_reflex3['answer'].content}")
    logs.append(results_reflex3['subtask_desc'])

    cot_instruction4 = "Subtask 4: Analyze the fourth reaction: stirring product 3 with Pd/C under hydrogen atmosphere, and determine the structure of product 4."
    results4_cot = await self.cot(
        subtask_id="subtask_4",
        cot_instruction=cot_instruction4,
        input_list=[taskInfo, results_reflex3['thinking'], results_reflex3['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query", results_reflex3['thinking'], results_reflex3['answer']]
    )
    cot_reflect_instruction4 = "Subtask 4 Reflexion: Based on CoT output, review and refine the structure determination of product 4."
    critic_instruction4 = "Please review the validity and limitations of the proposed structure for product 4."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4, 'input': [taskInfo, results_reflex3['thinking'], results_reflex3['answer'], results4_cot['thinking'], results4_cot['answer']], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query", results_reflex3['thinking'], results_reflex3['answer'], results4_cot['thinking'], results4_cot['answer']]
    }
    critic_desc4 = {
        'instruction': critic_instruction4, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    results_reflex4 = await self.reflexion(
        subtask_id="subtask_4_reflexion",
        cot_reflect_desc=cot_reflect_desc4,
        critic_desc=critic_desc4,
        n_repeat=self.max_round
    )
    agents.append(f"CoT agent {results4_cot['cot_agent'].id}, analyzing fourth reaction, thinking: {results4_cot['thinking'].content}; answer: {results4_cot['answer'].content}")
    for i in range(self.max_round):
        agents.append(f"Reflexion CoT agent {results_reflex4['cot_agent'].id}, round {i}, refining product 4 structure, thinking: {results_reflex4['list_thinking'][i].content}; answer: {results_reflex4['list_answer'][i].content}")
        agents.append(f"Critic agent {results_reflex4['critic_agent'].id}, round {i}, feedback: {results_reflex4['list_feedback'][i].content}; correctness: {results_reflex4['list_correct'][i].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results_reflex4['thinking'].content}; answer - {results_reflex4['answer'].content}")
    logs.append(results_reflex4['subtask_desc'])

    debate_instruction5 = "Subtask 5: Based on the output of Subtask 4, compare the deduced structure of product 4 with the given multiple-choice options and select the correct answer choice (A, B, C, or D)."
    final_decision_instruction5 = "Subtask 5: Make final decision on the correct answer choice for product 4's structure."
    debate_desc5 = {
        "instruction": debate_instruction5,
        "context": ["user query", results_reflex4['thinking'], results_reflex4['answer']],
        "input": [taskInfo, results_reflex4['thinking'], results_reflex4['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc5 = {
        "instruction": final_decision_instruction5,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        final_decision_desc=final_decision_desc5,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results5['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, debating answer choice, thinking: {results5['list_thinking'][round][idx].content}; answer: {results5['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, calculating final answer, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
