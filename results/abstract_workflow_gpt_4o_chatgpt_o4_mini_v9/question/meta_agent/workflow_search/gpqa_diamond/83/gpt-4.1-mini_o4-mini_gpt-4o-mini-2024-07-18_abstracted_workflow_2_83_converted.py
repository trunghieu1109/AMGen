async def forward_83(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and classify key mathematical and computational elements

    cot_instruction_1 = (
        "Sub-task 1: Analyze the problem context of solving higher dimensional heat equations using higher order finite difference approximations and parallel splitting, "
        "with matrix exponential approximated by fractional approximation. Identify and classify key mathematical and computational elements involved, such as the heat equation characteristics, finite difference schemes, matrix exponential function, fractional approximation, and parallel splitting method. "
        "Use the given task information as context."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing problem context, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Extract and clarify the role and properties of matrix exponential fractional approximation in parallel algorithm design

    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, extract and clarify the role and properties of the matrix exponential function approximation by fractional approximation in the context of parallel algorithm design for the heat equation solver. "
        "Understand how fractional approximation affects the algorithm structure and parallelization potential."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}

    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, clarifying fractional approximation role, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2

    # Choose the most consistent answer by frequency
    answer2_counter = Counter(possible_answers_2)
    answer2_most_common = answer2_counter.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_most_common]
    answer2 = answermapping_2[answer2_most_common]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Analyze the four given choices in relation to problem context and fractional approximation role

    cot_instruction_3 = (
        "Sub-task 3: Analyze the four given choices: Existence of nonlocal boundary conditions, Stability analysis, Linear partial fraction of fractional approximation, Complex roots of fractional approximation, "
        "in relation to the problem context and the role of fractional approximation in parallelizing the algorithm. Identify their relevance and implications."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing choices relevance, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Evaluate and synthesize the impact of choices on parallelization

    # Sub-task 4: Evaluate relationship between linear partial fraction decomposition and parallel algorithm conversion

    cot_instruction_4 = (
        "Sub-task 4: Evaluate the relationship between the linear partial fraction decomposition of the fractional approximation and its impact on converting a sequential algorithm into a parallel algorithm, "
        "considering the mathematical and computational implications identified in previous subtasks."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluating linear partial fraction impact, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Assess relevance of stability analysis and nonlocal boundary conditions to parallelization

    debate_instruction_5 = (
        "Sub-task 5: Assess the relevance of stability analysis and nonlocal boundary conditions to the parallelization process of the heat equation solver, "
        "based on the problem context and the nature of fractional approximation and matrix exponential function."
    )
    debate_roles = ["Pro-Stability", "Pro-Nonlocal", "Neutral"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3, answer3], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, assessing stability and boundary conditions, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the relevance of stability analysis and nonlocal boundary conditions to parallelization.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding on stability and boundary conditions relevance, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Determine significance of complex roots of fractional approximation in parallel splitting and algorithm conversion

    cot_instruction_6 = (
        "Sub-task 6: Determine the significance of complex roots of fractional approximation in the context of parallel splitting and algorithm conversion from sequential to parallel, "
        "integrating insights from previous subtasks."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking3, answer3, thinking4, answer4], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, analyzing complex roots significance, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Sub-task 7: Synthesize findings to identify key factor enabling conversion to parallel algorithm

    cot_reflect_instruction_7 = (
        "Sub-task 7: Synthesize findings from evaluation of linear partial fraction decomposition, stability analysis, nonlocal boundary conditions, and complex roots of fractional approximation "
        "to identify the key factor enabling the conversion of the sequential algorithm into a parallel algorithm in the given problem context."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round

    cot_inputs_7 = [taskInfo, thinking4, answer4, thinking5, answer5, thinking6, answer6]

    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, synthesizing key factor, thinking: {thinking7.content}; answer: {answer7.content}")

    for i in range(N_max_7):
        feedback, correct = await critic_agent_7([taskInfo, thinking7, answer7],
                                                "Please review the synthesis of the key factor enabling parallelization and provide its limitations.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining synthesis, thinking: {thinking7.content}; answer: {answer7.content}")

    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Subtask 7 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
