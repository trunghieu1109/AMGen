async def forward_19(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Sub-task 1: Extract assumptions
    cot_instruction1 = "Sub-task 1: From the query, extract and list the four assumptions numbered 1 through 4, preserving exact wording."
    cot_agent1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, extracting assumptions, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1:", sub_tasks[-1])

    # Sub-task 2: Extract multiple-choice options
    cot_instruction2 = "Sub-task 2: Extract and list the four multiple-choice answer options with their corresponding sets of assumptions from the query."
    cot_agent2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent2([taskInfo, thinking1, answer1], cot_instruction2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, extracting multiple-choice options, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2:", sub_tasks[-1])

    # Sub-task 3: Define impulse approximation
    cot_instruction3 = "Sub-task 3: Define the impulse approximation in the context of electroweak interactions in many-body nuclear calculations."
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, defining impulse approximation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3:", sub_tasks[-1])

    # Sub-task 4: Evaluate assumption 1
    cot_instruction4 = "Sub-task 4: Using the definition from Sub-task 3, evaluate whether assumption 1 ('The interaction current only interacts with individual nucleons') is necessary and consistent with the impulse approximation."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, evaluating assumption 1, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4:", sub_tasks[-1])

    # Sub-task 5: Evaluate assumption 2
    cot_instruction5 = "Sub-task 5: Using the definition from Sub-task 3, evaluate whether assumption 2 ('The nucleus is transparent apart from the selected nucleon') is necessary and consistent with the impulse approximation."
    cot_agent5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent5([taskInfo, thinking3, answer3], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, evaluating assumption 2, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5:", sub_tasks[-1])

    # Sub-task 6: Evaluate assumption 3
    cot_instruction6 = "Sub-task 6: Using the definition from Sub-task 3, evaluate whether assumption 3 ('The quarks internal to the selected nucleon are non-relativistic') is necessary and consistent with the impulse approximation."
    cot_agent6 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent6([taskInfo, thinking3, answer3], cot_instruction6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, evaluating assumption 3, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6:", sub_tasks[-1])

    # Sub-task 7: Evaluate assumption 4
    cot_instruction7 = "Sub-task 7: Using the definition from Sub-task 3, evaluate whether assumption 4 ('The interaction proceeds as if the selected nucleon experiences no binding forces') is necessary and consistent with the impulse approximation."
    cot_agent7 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent7([taskInfo, thinking3, answer3], cot_instruction7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent7.id}, evaluating assumption 4, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7:", sub_tasks[-1])

    # Sub-task 8: Combine results
    reflect_instruction8 = "Sub-task 8: Combine the necessity and consistency evaluations from Sub-tasks 4–7 to identify which subset of assumptions jointly imply the impulse approximation."
    cot_agent8 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent8 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    co_inputs8 = [taskInfo, thinking4, answer4, thinking5, answer5, thinking6, answer6, thinking7, answer7]
    thinking8, answer8 = await cot_agent8(co_inputs8, reflect_instruction8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent8.id}, combining results, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(N_max):
        feedback8, correct8 = await critic_agent8([taskInfo, thinking8, answer8], "Please evaluate the combined subset selection for completeness and correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent8.id}, feedback, thinking: {feedback8.content}; answer: {correct8.content}")
        if correct8.content == "True":
            break
        co_inputs8.extend([thinking8, answer8, feedback8])
        thinking8, answer8 = await cot_agent8(co_inputs8, reflect_instruction8, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent8.id}, refining combination, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8:", sub_tasks[-1])

    # Sub-task 9: Select correct choice
    debate_instruction9 = "Sub-task 9: Compare the subset identified in Sub-task 8 with the extracted multiple-choice options and select the correct choice number."
    debate_agents9 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max9 = self.max_round
    all_thinking9 = [[] for _ in range(N_max9)]
    all_answer9 = [[] for _ in range(N_max9)]
    for r in range(N_max9):
        for i, agent in enumerate(debate_agents9):
            inputs9 = [taskInfo, thinking2, answer2, thinking8, answer8]
            if r > 0:
                inputs9 = inputs9 + all_thinking9[r-1] + all_answer9[r-1]
            thinking9, answer9 = await agent(inputs9, debate_instruction9, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting choice, thinking: {thinking9.content}; answer: {answer9.content}")
            all_thinking9[r].append(thinking9)
            all_answer9[r].append(answer9)
    final_decision9 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision9([taskInfo] + all_thinking9[-1] + all_answer9[-1], "Sub-task 9: Make final decision on the correct choice.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision9.id}, decision, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Step 9:", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer