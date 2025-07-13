async def forward_7(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    debate_instr = "Sub-task 1: Translate the given logarithmic equations log_x(y^x) = 10 and log_y(x^{4y}) = 10 into explicit algebraic equations involving x and y. Carefully apply logarithm properties and exponent rules, ensuring domain conditions x > 1 and y > 1 are noted for validity. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking, answer = await agent([taskInfo], debate_instr, r, is_sub_task=True)
            else:
                input_infos = [taskInfo] + all_thinking[r-1] + all_answer[r-1]
                thinking, answer = await agent(input_infos, debate_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, translating logarithmic equations, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    final_decision_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], "Sub-task 1: Final decision on algebraic translation." + final_decision_instr, is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing algebraic translation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction = "Sub-task 2: Based on the algebraic equations from Sub-task 1, solve the system for real numbers x and y greater than 1. Perform algebraic manipulations, isolate variables, substitute expressions, and respect domain constraints. Consider multiple solution paths independently to ensure correctness and efficiency."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    possible_thinkings = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, solving system of equations, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2)
        possible_thinkings.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings + possible_answers, "Sub-task 2: Synthesize and choose the most consistent and correct solution." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 3: Verify the solutions obtained in Sub-task 2 against the original logarithmic equations and domain conditions. Compute the product xy as required. Include a reflexion step to confirm no logical or computational errors remain, thereby preventing inefficiencies or mistakes encountered previously." + reflect_inst
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction,
        "context": ["user query", thinking1, answer1, thinking2, answer2],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, verifying and refining solution, thinking: {thinking3.content}; answer: {answer3.content}")
    critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], "Please review and provide the limitations of provided solutions." + critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining solution, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
