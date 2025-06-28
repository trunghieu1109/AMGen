async def forward_155(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and classify elements

    # Sub-task 1: Analyze the starting materials (E)-oct-4-ene and (Z)-oct-4-ene
    cot_instruction_1 = (
        "Sub-task 1: Analyze the starting materials (E)-oct-4-ene and (Z)-oct-4-ene to identify their stereochemical configurations "
        "and relevant structural features that influence their reactivity with mCPBA."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing starting materials, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Analyze reagent mCPBA and reaction conditions
    cot_instruction_2 = (
        "Sub-task 2: Analyze the reagent mCPBA and the reaction conditions (one equivalent, followed by aqueous acid) "
        "to understand the type of reaction occurring (epoxidation) and the stereochemical outcome expected for each alkene isomer."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing reagent and conditions, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Determine stereochemical outcome of epoxidation of (E)-oct-4-ene
    cot_instruction_3 = (
        "Sub-task 3: Determine the stereochemical outcome of the epoxidation of (E)-oct-4-ene with mCPBA and aqueous acid, "
        "including the number and nature of stereoisomeric epoxide products formed."
    )
    N_sc = self.max_sc if hasattr(self, 'max_sc') else 5
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, determining stereochemical outcome of (E)-oct-4-ene epoxidation, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose most frequent answer
    from collections import Counter
    counter_3 = Counter(possible_answers_3)
    answer3_final = counter_3.most_common(1)[0][0]
    thinking3_final = thinkingmapping_3[answer3_final]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Determine stereochemical outcome of epoxidation of (Z)-oct-4-ene
    cot_instruction_4 = (
        "Sub-task 4: Determine the stereochemical outcome of the epoxidation of (Z)-oct-4-ene with mCPBA and aqueous acid, "
        "including the number and nature of stereoisomeric epoxide products formed."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking2, answer2], cot_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, determining stereochemical outcome of (Z)-oct-4-ene epoxidation, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    counter_4 = Counter(possible_answers_4)
    answer4_final = counter_4.most_common(1)[0][0]
    thinking4_final = thinkingmapping_4[answer4_final]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4_final.content}; answer - {answer4_final}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 2: Combine and analyze chromatographic behavior

    # Sub-task 5: Combine products and analyze total number of distinct stereoisomeric products
    debate_roles = ["Agent A", "Agent B"]
    debate_instruction_5 = (
        "Sub-task 5: Combine the products from the epoxidation of (E)- and (Z)-oct-4-ene and analyze the total number of distinct stereoisomeric products present in the mixture."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round if hasattr(self, 'max_round') else 3
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3_final, answermapping_3[answer3_final], thinking4_final, answermapping_4[answer4_final]], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3_final, answermapping_3[answer3_final], thinking4_final, answermapping_4[answer4_final]] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, combining products and analyzing stereoisomers, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on total number of distinct stereoisomeric products.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding total stereoisomeric products, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Evaluate expected chromatographic behavior on standard (achiral) reverse-phase HPLC
    cot_instruction_6 = (
        "Sub-task 6: Evaluate the expected chromatographic behavior of the combined product mixture on a standard (achiral) reverse-phase HPLC column, "
        "considering the number of distinct peaks that would be resolved given maximum theoretical resolution."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, evaluating standard HPLC behavior, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Sub-task 7: Evaluate expected chromatographic behavior on chiral HPLC
    cot_instruction_7 = (
        "Sub-task 7: Evaluate the expected chromatographic behavior of the combined product mixture on a chiral HPLC column, "
        "considering the number of distinct peaks that would be resolved given maximum theoretical resolution."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking5, answer5], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, evaluating chiral HPLC behavior, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Subtask 7 answer: ", sub_tasks[-1])

    # Sub-task 8: Compare chromatographic results and determine best matching answer choice
    cot_reflect_instruction_8 = (
        "Sub-task 8: Compare the chromatographic results from the standard and chiral HPLC analyses to determine which of the provided answer choices "
        "(4 peaks in both, 3 peaks standard/4 peaks chiral, 2 peaks standard/3 peaks chiral, or 2 peaks both) best matches the expected observations."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking6, answer6, thinking7, answer7], cot_reflect_instruction_8, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_8.id}, comparing chromatographic results and selecting best answer, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Subtask 8 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer
