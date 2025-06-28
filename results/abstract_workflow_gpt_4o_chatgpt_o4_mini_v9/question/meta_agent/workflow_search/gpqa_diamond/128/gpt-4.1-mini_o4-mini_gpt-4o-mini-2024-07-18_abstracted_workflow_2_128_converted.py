async def forward_128(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements
    # Sub-task 1: Analyze the reaction scheme step-by-step to identify chemical transformations
    cot_instruction_1 = (
        "Sub-task 1: Analyze the given reaction scheme step-by-step to identify the chemical transformations "
        "from compound A to compound E, including reagents and conditions used, to understand the nature of each intermediate compound."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing reaction scheme step-by-step, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze structural info and spectral data of compound A
    cot_instruction_2 = (
        "Sub-task 2: Analyze the structural information and spectral data provided for compound A, "
        "including the IR peak at 1750 cm^-1 and the reaction with an unknown phosphorous ylide forming 1,2-dimethyl-4-(propan-2-ylidene)cyclopentane, "
        "to deduce the likely functional groups and ring size of compound A."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing compound A spectral and structural data, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Integrate info from subtask 1 and 2 to hypothesize structure of compound A using Reflexion
    cot_reflect_instruction_3 = (
        "Sub-task 3: Based on outputs from Sub-task 1 and Sub-task 2, integrate the information to hypothesize the structure of compound A, "
        "considering the reaction with HCN, the phosphorous ylide reaction, and the IR data."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, hypothesizing structure of compound A, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3],
                                               "Please review the hypothesized structure of compound A and provide limitations or corrections.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining hypothesis of compound A, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Predict and Confirm Structures of B, C, D, E
    # Sub-task 4: Predict structures of B, C, D, E based on transformations and hypothesized A
    cot_instruction_4 = (
        "Sub-task 4: Predict the structures of compounds B, C, D, and E based on the transformations described (addition of HCN to A, reduction of B, reaction with HNO2, and elimination of inert gas from D), "
        "using the hypothesized structure of A from Sub-task 3."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, predicting structures of B, C, D, E, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Analyze IR spectral data of compound E and compare with A to confirm carbonyl group
    debate_instruction_5 = (
        "Sub-task 5: Analyze the IR spectral data of compound E (strong peak at 1715 cm^-1) and compare it with the IR data of compound A to confirm the presence and type of carbonyl group in compound E, "
        "supporting the predicted structure from Sub-task 4."
    )
    debate_roles = ["Spectroscopy Expert", "Organic Chemist"]
    debate_agents_5 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in debate_roles
    ]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing IR data of compound E, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1],
                                                     "Sub-task 5: Make final decision on the IR spectral analysis and confirmation of compound E's carbonyl group.",
                                                     is_sub_task=True)
    agents.append(f"Final Decision agent, analyzing IR data and confirming carbonyl group, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compare predicted structure of compound E with multiple-choice options to select correct compound
    cot_instruction_6 = (
        "Sub-task 6: Compare the predicted structure of compound E with the given multiple-choice options (4-methylcycloheptan-1-one, 2,3,4-trimethylcyclopentan-1-one, "
        "2,2,3,4-tetramethylcyclobutan-1-one, 3,4-dimethylcyclohexan-1-one) using all gathered information including ring size, substitution pattern, and IR data to select the correct compound E."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, selecting correct compound E from options, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
