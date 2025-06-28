async def forward_54(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []
    
    # Stage 0: Extract and summarize inputs
    
    # Sub-task 1: Extract and list all given 1H NMR spectral data points
    cot_instruction_1 = (
        "Sub-task 1: Extract and list all given 1H NMR spectral data points (chemical shifts, multiplicities, coupling constants, and integration) "
        "from the query to clearly understand the spectral features of the unknown compound."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracted NMR data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    
    # Sub-task 2: Summarize structural features and key differences of candidate compounds focusing on expected 1H NMR characteristics
    cot_instruction_2 = (
        "Sub-task 2: Summarize the structural features and key differences of the four candidate compounds (Trans-propenyl acetate, Cis-propenyl acetate, "
        "Trans-butenyl acetate, Cis-butenyl acetate) focusing on their expected 1H NMR characteristics, especially coupling constants and multiplicities."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, summarized candidate NMR features, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    
    # Stage 1: Analyze coupling constants and multiplicities, and determine proton types
    
    # Sub-task 3: Analyze coupling constants and multiplicities to infer alkene geometry and substituents
    cot_sc_instruction_3 = (
        "Sub-task 3: Analyze the coupling constants and multiplicities from the extracted NMR data (Sub-task 1) to infer the type of alkene geometry (cis or trans) "
        "and the presence of substituents, using the summarized expected NMR features of candidate compounds (Sub-task 2)."
    )
    N_sc = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, analyzed coupling constants and multiplicities, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose the most consistent answer
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3_content = thinkingmapping_3[answer3_content].content
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_content}; answer - {answer3_content}")
    print("Step 3: ", sub_tasks[-1])
    
    # Sub-task 4: Determine number and type of protons from integration and multiplicity to distinguish propenyl vs butenyl
    cot_sc_instruction_4 = (
        "Sub-task 4: Determine the number and type of protons (e.g., vinyl, methyl, methylene) from the integration and multiplicity data to distinguish between propenyl and butenyl acetate structures, "
        "referencing the candidate compound features (Sub-task 2)."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, analyzed proton types and integration, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4_content = thinkingmapping_4[answer4_content].content
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4_content}; answer - {answer4_content}")
    print("Step 4: ", sub_tasks[-1])
    
    # Stage 2: Combine inferences and select the most likely compound
    
    # Sub-task 5: Combine geometric inference and alkyl chain length inference to identify the most likely compound
    cot_reflect_instruction_5 = (
        "Sub-task 5: Combine the geometric inference (cis/trans) from coupling constants (Sub-task 3) and the alkyl chain length inference (propenyl vs butenyl) from proton count and multiplicity (Sub-task 4) "
        "to identify the most likely compound among the four choices."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking3, answermapping_3[answer3_content], thinking4, answermapping_4[answer4_content]]
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, combined inferences, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5],
                                                "Critically evaluate the combined inference and identify any inconsistencies or missing considerations.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining combined inference, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    
    # Sub-task 6: Map identified features and NMR interpretation results to given answer choices and select correct compound name
    debate_instruction_6 = (
        "Sub-task 6: Based on the identified structural features and NMR interpretation results (Sub-task 5), map to the given answer choices and select the correct compound name."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            input_infos_6 = [taskInfo, thinking5, answer5]
            if r > 0:
                input_infos_6 += all_thinking6[r-1] + all_answer6[r-1]
            thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, mapping to answer choices, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the correct compound name.", is_sub_task=True)
    agents.append(f"Final Decision agent on selecting compound, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
