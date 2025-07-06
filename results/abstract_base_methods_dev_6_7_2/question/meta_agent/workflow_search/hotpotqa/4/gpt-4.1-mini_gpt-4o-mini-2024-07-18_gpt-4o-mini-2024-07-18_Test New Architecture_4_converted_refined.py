async def forward_4(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    magazine_names = ["Woman's Era", "Naj"]
    magazine_types = []

    cot_reflect_instruction1 = "Sub-task 1: Research and identify the official category or type of the magazine 'Woman's Era' by citing authoritative sources or official publisher descriptions. Provide evidence and reflect on the confidence level of this classification."
    critic_instruction1 = "Please review the classification of 'Woman's Era' magazine, check for unsupported claims or missing evidence, and provide feedback."
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo, "Woman's Era"],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "magazine name: Woman's Era"]
    }
    critic_desc1 = {
        'instruction': critic_instruction1,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results1 = await self.reflexion(
        subtask_id="subtask_1",
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, classifying 'Woman's Era', thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback on 'Woman's Era' classification, thinking: {results1['list_feedback'][i].content}; correct: {results1['list_correct'][i].content}")
        if i + 1 < len(results1['list_thinking']) and i + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining 'Woman's Era' classification, thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    magazine_types.append(results1['answer'].content)

    cot_reflect_instruction2 = "Sub-task 2: Independently research and identify the official category or type of the magazine 'Naj' by citing authoritative sources or official publisher descriptions. Provide evidence and reflect on the confidence level of this classification."
    critic_instruction2 = "Please review the classification of 'Naj' magazine, check for unsupported claims or missing evidence, and provide feedback."
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, "Naj"],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "magazine name: Naj"]
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
    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, classifying 'Naj', thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results2['list_feedback']))):
        agents.append(f"Critic agent {results2['critic_agent'].id}, feedback on 'Naj' classification, thinking: {results2['list_feedback'][i].content}; correct: {results2['list_correct'][i].content}")
        if i + 1 < len(results2['list_thinking']) and i + 1 < len(results2['list_answer']):
            agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining 'Naj' classification, thinking: {results2['list_thinking'][i + 1].content}; answer: {results2['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    magazine_types.append(results2['answer'].content)

    debate_instruction3 = "Sub-task 3: Critically compare and contrast the classifications of 'Woman's Era' and 'Naj' magazines from Sub-tasks 1 and 2. Identify any inconsistencies or uncertainties, and synthesize a coherent, justified final classification for both magazines."
    final_decision_instruction3 = "Sub-task 3: Make a final decision on the integrated classification of 'Woman's Era' and 'Naj' magazines, highlighting any doubts or limitations."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", f"thinking of subtask 1", f"answer of subtask 1", f"thinking of subtask 2", f"answer of subtask 2"],
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc3 = {
        'instruction': final_decision_instruction3,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        final_decision_desc=final_decision_desc3,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results3['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, comparing magazine classifications, thinking: {results3['list_thinking'][round][idx].content}; answer: {results3['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, synthesizing final classification, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_reflect_instruction4 = "Sub-task 4: Fact-check the integrated classification of 'Woman's Era' and 'Naj' magazines against authoritative sources. Identify any inaccuracies or missing information, and revise the final answer accordingly. Explicitly state any uncertainties or limitations."
    critic_instruction4 = "Please review the fact-checking and revision of the integrated magazine classification, providing feedback on accuracy and completeness."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    critic_desc4 = {
        'instruction': critic_instruction4,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results4 = await self.reflexion(
        subtask_id="subtask_4",
        cot_reflect_desc=cot_reflect_desc4,
        critic_desc=critic_desc4,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, fact-checking integrated classification, thinking: {results4['list_thinking'][0].content}; answer: {results4['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results4['list_feedback']))):
        agents.append(f"Critic agent {results4['critic_agent'].id}, feedback on fact-checking, thinking: {results4['list_feedback'][i].content}; correct: {results4['list_correct'][i].content}")
        if i + 1 < len(results4['list_thinking']) and i + 1 < len(results4['list_answer']):
            agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining final answer, thinking: {results4['list_thinking'][i + 1].content}; answer: {results4['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs
