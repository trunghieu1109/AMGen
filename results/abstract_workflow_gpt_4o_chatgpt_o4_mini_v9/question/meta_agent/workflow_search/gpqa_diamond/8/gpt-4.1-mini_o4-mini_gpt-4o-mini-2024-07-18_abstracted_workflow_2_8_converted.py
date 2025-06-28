async def forward_8(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Identify and collect structural and symmetry-related information for each molecule
    cot_instruction_1 = (
        "Sub-task 1: Identify and collect structural and symmetry-related information for each molecule: quinuclidine, triisopropyl borate, "
        "benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone. "
        "Use reliable chemical databases or literature to understand their molecular geometry and symmetry elements."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, collecting structural and symmetry info, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Stage 1: Analyze molecular structures to determine point group symmetry focusing on C3h
    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze the molecular structures obtained in Sub-task 1 to determine the point group symmetry of each molecule, "
        "with emphasis on identifying presence or absence of C3h symmetry elements. Consider all possible symmetry elements and configurations."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining point group symmetry focusing on C3h, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Aggregate most consistent answer
    answer_counter_2 = Counter(possible_answers_2)
    most_common_answer_2 = answer_counter_2.most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[most_common_answer_2]
    answer2_final = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Compare symmetry characteristics to identify molecule(s) with C3h symmetry
    cot_instruction_3 = (
        "Sub-task 3: Compare the symmetry characteristics of all four molecules based on Sub-task 2 analysis to identify which molecule(s) exhibit C3h symmetry. "
        "Provide detailed comparison and reasoning."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2_final, answer2_final], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, comparing symmetry characteristics, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Validate identified molecule(s) with C3h symmetry by cross-referencing authoritative sources
    cot_reflect_instruction_4 = (
        "Sub-task 4: Validate the identified molecule(s) with C3h symmetry by cross-referencing with authoritative chemical symmetry references or computational symmetry analysis tools to confirm accuracy. "
        "Critically evaluate and refine the identification."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, validating C3h symmetry identification, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4],
                                               "Please review the validation of C3h symmetry identification and provide limitations or confirm correctness.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining validation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
