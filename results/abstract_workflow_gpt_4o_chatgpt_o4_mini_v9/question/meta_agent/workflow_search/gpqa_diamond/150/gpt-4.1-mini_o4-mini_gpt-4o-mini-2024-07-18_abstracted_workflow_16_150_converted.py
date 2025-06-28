async def forward_150(self, taskInfo):
    from collections import Counter
    import numpy as np
    import math

    print("Task Requirement: ", taskInfo)

    sub_tasks = []
    agents = []

    # Stage 1: Normalize state vector and express observable operator P as matrix

    # Sub-task 1: Normalize the given state vector (-1, 2, 1)
    cot_instruction_1 = (
        "Sub-task 1: Normalize the given state vector at time t, which is the column matrix with elements (-1, 2, 1). "
        "Ensure the state vector is a valid quantum state with unit norm before further calculations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, normalizing state vector, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Express observable operator P as matrix
    cot_instruction_2 = (
        "Sub-task 2: Express the observable operator P as a matrix using the given elements: first row (0, 1/sqrt(2), 0), "
        "second row (1/sqrt(2), 0, 1/sqrt(2)), and third row (0, 1/sqrt(2), 0). "
        "Prepare the operator for eigenvalue and eigenvector analysis."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, expressing observable operator P, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Find eigenvalues and eigenvectors, identify eigenvector(s) for eigenvalue 0

    # Sub-task 3: Find eigenvalues and eigenvectors of P
    cot_instruction_3 = (
        "Sub-task 3: Find the eigenvalues and corresponding eigenvectors of the observable operator P matrix constructed in Sub-task 2. "
        "Identify the possible measurement outcomes and their associated states."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Self-Consistency Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    N = self.max_sc if hasattr(self, 'max_sc') else 5
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Self-Consistency Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, finding eigenvalues and eigenvectors, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose the most consistent answer
    most_common_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[most_common_answer_3]
    answer3 = answermapping_3[most_common_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Identify eigenvector(s) corresponding to eigenvalue 0
    cot_instruction_4 = (
        "Sub-task 4: Identify the eigenvector(s) corresponding to the eigenvalue 0 from the eigenvalues and eigenvectors found in Sub-task 3. "
        "This eigenvector represents the state(s) associated with the measurement outcome 0."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, identifying eigenvector(s) for eigenvalue 0, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 3: Project normalized state onto eigenvector(s) and calculate probability

    # Sub-task 5: Project normalized state vector onto eigenvector(s) for eigenvalue 0
    debate_instruction_5 = (
        "Sub-task 5: Project the normalized state vector from Sub-task 1 onto the eigenvector(s) corresponding to eigenvalue 0 found in Sub-task 4. "
        "Calculate the component of the system's state aligned with the measurement outcome 0."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round if hasattr(self, 'max_round') else 3
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4, thinking1, answer1], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4, thinking1, answer1] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, projecting state vector, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    # Final decision agent synthesizes projection
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the projection component.", is_sub_task=True)
    agents.append(f"Final Decision agent, projecting normalized state vector, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Calculate probability of measurement yielding 0
    cot_reflect_instruction_6 = (
        "Sub-task 6: Calculate the probability that the measurement of the observable yields 0 by computing the squared magnitude of the projection obtained in Sub-task 5. "
        "This gives the probability of measuring the eigenvalue 0 at time t."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_reflect_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, calculating probability of measurement 0, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
