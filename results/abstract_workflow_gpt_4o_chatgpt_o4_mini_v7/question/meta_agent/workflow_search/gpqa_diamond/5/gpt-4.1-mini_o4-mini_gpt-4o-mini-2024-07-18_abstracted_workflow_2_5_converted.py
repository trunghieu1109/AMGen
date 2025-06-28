async def forward_5(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and classify the potential function
    cot_instruction_1 = (
        "Sub-task 1: Analyze the given potential function V(r, θ) = 1/2 kr^2 + 3/2 kr^2 cos^2(θ), "
        "identify its mathematical form, symmetries, and classify it in terms of known quantum mechanical potentials. "
        "Rewrite the potential in a more convenient form if necessary."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                               model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing potential function, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Stage 1: Express the potential in Cartesian coordinates
    cot_sc_instruction_2 = (
        "Sub-task 2: Using the output from Sub-task 1, express the potential V(r, θ) in Cartesian coordinates (x, y) "
        "using x = r cos(θ), y = r sin(θ), to identify effective frequencies along x and y directions."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                 model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}

    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, expressing potential in Cartesian coordinates, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2

    # Choose the most consistent answer by majority vote
    answer2_final = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[answer2_final]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Identify effective harmonic oscillator frequencies ω_x and ω_y
    cot_instruction_3 = (
        "Sub-task 3: From the Cartesian form of the potential, identify the effective harmonic oscillator frequencies ω_x and ω_y "
        "along x and y directions by comparing terms to the standard form 1/2 m ω^2 x^2."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                               model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2_final, answer2_final], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, identifying effective frequencies, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Write down the Hamiltonian and recall energy eigenvalues formula
    cot_instruction_4 = (
        "Sub-task 4: Write the Hamiltonian for the 2D quantum harmonic oscillator with frequencies ω_x and ω_y, "
        "and recall the general formula for the energy eigenvalues of the anisotropic oscillator."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                               model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, writing Hamiltonian and recalling energy formula, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 2: Derive explicit expression for energy spectrum E(n_x, n_y)
    cot_instruction_5 = (
        "Sub-task 5: Derive the explicit expression for the energy spectrum E(n_x, n_y) in terms of quantum numbers n_x, n_y, ℏ, m, and k, "
        "using the frequencies ω_x and ω_y found previously."
    )
    debate_roles = ["Physicist A", "Physicist B"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in debate_roles]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], 
                                               cot_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, cot_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, deriving energy spectrum, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 
                                                     "Sub-task 5: Make final decision on the explicit energy spectrum expression.", 
                                                     is_sub_task=True)
    agents.append(f"Final Decision agent, deciding energy spectrum, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Stage 2: Compare derived energy spectrum with given multiple-choice options
    cot_instruction_6 = (
        "Sub-task 6: Compare the derived energy spectrum expression with the given multiple-choice options "
        "to identify which choice matches the correct energy spectrum for the given potential."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                               model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, comparing energy spectrum with options, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
