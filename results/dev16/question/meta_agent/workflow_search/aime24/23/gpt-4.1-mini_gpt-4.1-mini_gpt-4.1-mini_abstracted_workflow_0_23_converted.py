async def forward_23(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Stage 0 Sub-task 1: Define variables representing each digit in the 2x3 grid
    cot_instruction = (
        "Sub-task 1: Define variables d11, d12, d13 for the top row digits and d21, d22, d23 for the bottom row digits. "
        "Express the two row numbers as concatenations d11d12d13 and d21d22d23 (left to right). "
        "Express the three column numbers as concatenations of top digit then bottom digit for each column. "
        "Each cell contains a single digit from 0 to 9."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, defining variables, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Stage 0 Sub-task 2: Clarify assumptions about digit ranges, leading zeros, and piecewise column number definition
    cot_sc_instruction = (
        "Sub-task 2: Based on variable definitions, clarify and explicitly state all assumptions: digits 0-9, leading zeros allowed in row and column numbers, "
        "fixed reading order (rows left to right, columns top to bottom). "
        "Critically, define column numbers piecewise: if top digit d1j=0, column number equals bottom digit d2j (single-digit); else column number equals 10*d1j + d2j (two-digit). "
        "Avoid assuming all column numbers are two-digit."
    )
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    possible_thinkings = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, clarifying assumptions, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2)
        possible_thinkings.append(thinking2)
    final_instr = "Given all the above thinking and answers, synthesize the most consistent and correct assumptions about digit ranges and column number formation."
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_thinkings, "Sub-task 2: Synthesize assumptions." + final_instr, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Stage 0 Sub-task 3: Formulate sum constraints algebraically with piecewise column number definition
    cot_instruction = (
        "Sub-task 3: Formulate algebraic sum constraints using variables d11,d12,d13,d21,d22,d23. "
        "Row sum: (100*d11 + 10*d12 + d13) + (100*d21 + 10*d22 + d23) = 999. "
        "Column sum: sum over j=1 to 3 of column numbers, where column number = d2j if d1j=0 else 10*d1j + d2j, equals 99. "
        "Introduce indicator variables z_j = 1 if d1j=0 else 0 to represent piecewise cases. "
        "Do not restrict sums s = d11+d12+d13 and t = d21+d22+d23 to single digits."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction,
        "context": ["user query", thinking2.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent([taskInfo, thinking2], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, formulating algebraic constraints, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Stage 0 Sub-task 4: Solve column sum equation 10*s + t = 99 with full digit sum ranges
    cot_instruction = (
        "Sub-task 4: Solve the equation 10*s + t = 99 where s = d11 + d12 + d13 and t = d21 + d22 + d23, "
        "with 0 <= s <= 27 and 0 <= t <= 27. Identify all integer solutions and record valid ones. "
        "Cross-check consistency with bottom row digits implied by row sum and digit relations. "
        "Avoid assuming s and t are single digits or ignoring carry possibilities."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction,
        "context": ["user query", thinking3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent([taskInfo, thinking3], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, solving column sum equation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Stage 0 Sub-task 5: Analyze algebraic constraints including carry-over with corrected sums s=8 and t=19
    cot_instruction = (
        "Sub-task 5: Analyze digit relationships and carry-over in addition for row and column sums, "
        "incorporating corrected sums s=8 and t=19. Model carry propagation explicitly, considering piecewise column number definitions and leading zeros. "
        "Avoid simplifying assumptions excluding carries or treating column sums as linear functions."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction,
        "context": ["user query", thinking4.content, thinking2.content],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent([taskInfo, thinking4, thinking2], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing carry-over and digit relations, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Stage 1 Sub-task 1: Combine row and column sum constraints with carry-over relations into digit-wise system
    cot_sc_instruction = (
        "Sub-task 1: Combine all constraints from previous subtasks into a system of digit-wise equations relating digits and carry variables. "
        "Explicitly incorporate piecewise column number definitions and indicator variables for zero top digits. "
        "Ensure system captures all dependencies and constraints without oversimplification."
    )
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    possible_thinkings = []
    subtask_desc6 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query", thinking5.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking6, answer6 = await cot_agents[i]([taskInfo, thinking5], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, combining constraints, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers.append(answer6)
        possible_thinkings.append(thinking6)
    final_instr = "Given all the above thinking and answers, synthesize the most consistent system of digit-wise equations and carry relations."
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + possible_thinkings, "Stage 1 Sub-task 1: Synthesize system." + final_instr, is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Stage 1 Sub-task 2: Transform system into explicit digit and carry constraints, enumerate feasible combinations
    cot_instruction = (
        "Sub-task 2: Transform the system of equations into explicit constraints on digits and carry values. "
        "Enumerate all feasible digit-carry combinations consistent with sums and piecewise column definitions. "
        "Use systematic case distinctions based on which columns have zero top digits. Avoid premature pruning or ignoring valid carry scenarios."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction,
        "context": ["user query", thinking6.content],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent([taskInfo, thinking6], cot_instruction, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, enumerating digit-carry combinations, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Stage 1 Sub-task 3: Aggregate constraints to reduce search space and prepare candidate digit assignments
    cot_instruction = (
        "Sub-task 3: Aggregate digit and carry constraints to reduce search space. "
        "Identify fixed digits, digit ranges, forced zero/nonzero conditions on top digits of columns. "
        "Prepare refined candidate digit assignments for enumeration."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction,
        "context": ["user query", thinking7.content],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent([taskInfo, thinking7], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, refining candidate digit assignments, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Stage 1 Sub-task 3 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Stage 2 Sub-task 1: Enumerate all digit assignments satisfying refined constraints and verify sums
    cot_instruction = (
        "Sub-task 1: Enumerate all digit assignments in the 2x3 grid satisfying refined constraints. "
        "For each candidate, compute row numbers and column numbers using piecewise definition, verify sums equal 999 and 99 respectively, and confirm all digit and carry constraints hold. "
        "Respect leading zeros and conditional column number formation."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc9 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query", thinking8.content],
        "agent_collaboration": "Reflexion"
    }
    thinking9, answer9 = await cot_agent([taskInfo, thinking8], cot_instruction, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, enumerating and verifying digit assignments, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Stage 2 Sub-task 2: Count total number of valid digit placements and verify correctness
    cot_instruction = (
        "Sub-task 2: Count total valid digit placements found in enumeration. "
        "Cross-check sums for each candidate, ensure no contradictions or invalid assumptions remain. "
        "Provide final validated count as answer."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc10 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction,
        "context": ["user query", thinking9.content],
        "agent_collaboration": "CoT"
    }
    thinking10, answer10 = await cot_agent([taskInfo, thinking9], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, counting valid digit placements, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Stage 2 Sub-task 2 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs
