async def forward_3(self, taskInfo):
    from collections import Counter

    print("Task Requirement: ", taskInfo)

    sub_tasks = []
    agents = []

    # Sub-task 1: List standard Maxwell's equations and their physical meanings without monopoles
    cot_instruction = "Sub-task 1: List the four standard Maxwell's equations in differential form without magnetic monopoles and briefly state what each one expresses physically in this context."
    cot_agent = LLMAgentBase(["thinking", "answer"], "CoT Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Classify each equation by operator and field using self-consistency CoT
    sc_instruction = "Sub-task 2: Based on the equations from Sub-task 1, classify each by its differential operator (divergence or curl), the field it acts on (E or B), and name the physical law it encodes."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "SC-CoT Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_map = {}
    answer_map = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], sc_instruction, is_sub_task=True)
        agents.append(f"SC-CoT agent {cot_agents[i].id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinking_map[answer2.content] = thinking2
        answer_map[answer2.content] = answer2
    most_common = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_map[most_common]
    answer2 = answer_map[most_common]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Modify equations to include magnetic charge and current densities with reflexion
    reflect_instruction = "Sub-task 3: Starting from the standard Maxwell's equations, modify them to include magnetic charge density rho_m and current density J_m, derive the new differential forms, and identify which equations gain extra source or current terms."
    cot_agent = LLMAgentBase(["thinking", "answer"], "CoT Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    N_max = self.max_round
    thinking3, answer3 = await cot_agent(cot_inputs, reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], 
            "Critically evaluate the modified equations for correctness and completeness and point out any missing source or current term errors.",
            i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Debate to select the correct answer choice
    debate_instruction = "Sub-task 4: Compare the modified Maxwell's equations from Sub-task 3 against the provided choices and select which choice correctly names the equations that change when magnetic monopoles are allowed."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds = self.max_round
    all_thinking4 = [[] for _ in range(N_rounds)]
    all_answer4 = [[] for _ in range(N_rounds)]
    for r in range(N_rounds):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                inputs = [taskInfo, thinking3, answer3]
            else:
                inputs = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
            thinking4, answer4 = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent([taskInfo] + all_thinking4[-1] + all_answer4[-1], 
        "Sub-task 4: Make final decision on which answer choice is correct.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer