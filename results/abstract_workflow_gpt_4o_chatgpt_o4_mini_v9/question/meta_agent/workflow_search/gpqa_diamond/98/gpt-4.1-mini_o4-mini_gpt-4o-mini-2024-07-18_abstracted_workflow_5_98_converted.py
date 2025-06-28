async def forward_98(self, taskInfo):
    from collections import Counter

    print("Task Requirement: ", taskInfo)

    sub_tasks = []
    agents = []

    # Stage 0: Extract and characterize spectral features
    # Sub-task 1: Extract FTIR features using Chain-of-Thought
    cot_instruction_1 = (
        "Sub-task 1: Extract and characterize the key spectral features from the FTIR data: "
        "Identify the significance of the very broad absorption peak at 3000 cm⁻¹ and the strong absorption peak at 1700 cm⁻¹, "
        "relating these to functional groups present in the unknown compound."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting FTIR features, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 0.1: ", sub_tasks[-1])

    # Sub-task 2: Extract 1H NMR features using Chain-of-Thought
    cot_instruction_2 = (
        "Sub-task 2: Extract and characterize the key features from the 1H NMR spectrum: "
        "Identify the number and types of peaks, note the absence of vinyl-hydrogens, and describe the complex splitting patterns such as the doublet of triplets of quartets and doublet of triplets of triplets."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, extracting 1H NMR features, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Analyze and classify spectral features
    # Sub-task 3: Analyze FTIR features to confirm functional groups using Self-Consistency Chain-of-Thought
    cot_sc_instruction_3 = (
        "Sub-task 3: Analyze the FTIR spectral features extracted in Sub-task 1 to classify the functional groups present, "
        "specifically confirming the presence of carboxylic acid (COOH) and other relevant groups based on the absorption peaks."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, analyzing FTIR features, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    most_common_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[most_common_answer_3]
    answer3 = answermapping_3[most_common_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 1.3: ", sub_tasks[-1])

    # Sub-task 4: Analyze 1H NMR features to classify hydrogen environments and splitting patterns using Self-Consistency Chain-of-Thought
    cot_sc_instruction_4 = (
        "Sub-task 4: Analyze the 1H NMR spectral features extracted in Sub-task 2 to classify the types of hydrogen environments and splitting patterns, "
        "and infer the possible neighboring groups and substituents on the molecule."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking2, answer2], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, analyzing 1H NMR features, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    most_common_answer_4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[most_common_answer_4]
    answer4 = answermapping_4[most_common_answer_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 1.4: ", sub_tasks[-1])

    # Stage 2: Compare inferred groups with candidate compounds
    # Sub-task 5: Compare functional groups and hydrogen environments with candidate compounds using Debate
    debate_instruction_5 = (
        "Sub-task 5: Compare the inferred functional groups and hydrogen environments from Sub-tasks 3 and 4 with the structural features of each candidate compound, "
        "focusing on the presence or absence of vinyl hydrogens, types of alkyl substituents (methyl vs ethyl), and the position of the carboxylic acid group."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3, answer3, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing candidates, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the candidate compound comparison.", is_sub_task=True)
    agents.append(f"Final Decision agent on candidate comparison, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 2.5: ", sub_tasks[-1])

    # Stage 3: Evaluate and prioritize candidate compounds
    # Sub-task 6: Evaluate and prioritize candidates using Reflexion
    cot_reflect_instruction_6 = (
        "Sub-task 6: Evaluate and prioritize the candidate compounds based on how well their predicted spectral features match the observed FTIR and 1H NMR data, "
        "including the complex splitting patterns and absence of vinyl hydrogens, to identify the most likely compound."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, evaluating candidates, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], "Please review the candidate evaluation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining candidate evaluation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 3.6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
