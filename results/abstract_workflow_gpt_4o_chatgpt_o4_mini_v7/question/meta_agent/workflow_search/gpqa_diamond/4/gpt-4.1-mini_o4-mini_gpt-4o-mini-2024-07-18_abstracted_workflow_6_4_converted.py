async def forward_4(self, taskInfo):
    from collections import Counter
    import cmath
    import math

    print("Task Requirement: ", taskInfo)

    sub_tasks = []
    agents = []

    # Stage 0: Analyze and extract operator components and direction vector

    # Sub-task 1: Analyze Px, Py, Pz matrices and scalar factors
    cot_instruction_1 = (
        "Sub-task 1: Analyze and extract the given quantum mechanical operator components Px, Py, and Pz matrices "
        "and their scalar factors (ħ/2), identify the operator P as a vector operator composed of these components, "
        "and confirm the basis (2x2 matrices) and physical context (spin-1/2 muon)."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing operator components, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Interpret direction vector n in x-z plane as (sinθ, 0, cosθ)
    cot_instruction_2 = (
        "Sub-task 2: Interpret the direction vector n lying in the x-z plane, parameterize it in terms of an angle θ, "
        "express n as (sinθ, 0, cosθ)."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, interpreting direction vector, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Construct operator P.n and set up eigenvalue equation

    # Sub-task 3: Construct operator P.n = Px*sinθ + Py*0 + Pz*cosθ
    cot_instruction_3 = (
        "Sub-task 3: Construct the operator P.n by taking the linear combination of Px, Py, and Pz weighted by the components "
        "of n (sinθ, 0, cosθ), using the matrices and ħ/2 factor from subtask_1 and the direction vector from subtask_2."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, constructing operator P.n, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Set up eigenvalue equation (P.n)|psi> = +ħ/2 |psi> explicitly as 2x2 matrix eigenvalue problem
    cot_instruction_4 = (
        "Sub-task 4: Set up the eigenvalue equation (P.n)|psi> = +ħ/2 |psi> using the operator from subtask_3, "
        "write it explicitly as a 2x2 matrix eigenvalue problem."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, setting up eigenvalue equation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Solve eigenvalue equation to find eigenvector |psi> for eigenvalue +ħ/2
    cot_instruction_5 = (
        "Sub-task 5: Solve the eigenvalue equation from subtask_4 to find the eigenvector |psi> corresponding to eigenvalue +ħ/2, "
        "solving the 2x2 linear algebra problem to find vector components."
    )
    debate_roles_5 = ["Proponent", "Opponent"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles_5]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], cot_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, cot_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, solving eigenvalue equation, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on eigenvector solution.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding eigenvector, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Normalize eigenvector and compare with choices

    # Sub-task 6: Normalize eigenvector from subtask_5
    cot_instruction_6 = (
        "Sub-task 6: Normalize the eigenvector obtained in subtask_5 to ensure unit norm, consistent with quantum mechanical state vector normalization."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, normalizing eigenvector, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Express normalized eigenvector in terms of θ and compare with given choices
    cot_instruction_7 = (
        "Sub-task 7: Express the normalized eigenvector in terms of θ (angle defining direction n in x-z plane), "
        "and compare the resulting form with the given multiple-choice options to identify the correct normalized eigenvector expression."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6, thinking2, answer2], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, expressing normalized eigenvector and comparing choices, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
